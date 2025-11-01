# fmt: off
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from math import ceil

import numpy as np
import torch
from tqdm import tqdm

if 'eval' in sys.path[0]:
    sys.path.insert(0, os.path.dirname(sys.path[0]))  # add mas folder to path

from data_loaders.base_dataset import collate
from data_loaders.dataset_utils import (
    get_dataset_loader,
    get_visualization_scale,
    sample_distance,
    sample_vertical_angle,
)
from eval.eval_motionBert import convert_motionBert_skeleton
from eval.metrics import (
    calculate_diversity,
    calculate_frechet_distance,
    calculate_precision,
    calculate_recall,
)
from sample.ablations import MAS_TYPES
from sample.generate import Generator
from sample.sampler import Sampler
from utils.fixseed import fixseed
from utils.math_utils import perspective_projection
from utils.parser_utils import evaluate_args

ELEPOSE_SAMPLES_PATH = "dataset/nba/elepose_predictions"
# MOTIONBERT_SAMPLES_PATH = "dataset/nba/motionBert_predictions"
MOTIONBERT_SAMPLES_PATH = "/home/galhar/gits/VideoMDM/dataset/nba/motionbert_predictions"
DIVERSITY_TIMES = 200
PRECISION_AND_RECALL_K = 3

SCALE_FOR_SAVED_SAMPLES = 2.20919891496

def interpolate_motion(motion, frames_to_interpolate):
    motion = torch.from_numpy(motion)
    motion_vel = motion[1:] - motion[:-1]
    motion_vel = torch.cat([motion_vel, torch.zeros((1, motion.shape[1], motion.shape[2]))], dim=0)
    motion_lengthened = torch.zeros((motion.shape[0] * frames_to_interpolate, motion.shape[1], motion.shape[2]))
    for i in range(frames_to_interpolate):
        motion_lengthened[i::frames_to_interpolate] = motion + (i / frames_to_interpolate) * motion_vel
    return motion_lengthened.numpy()

def convert_to_numpy(list_of_torch_tensors):
    return np.stack([tensor.cpu().numpy() for tensor in list_of_torch_tensors])


def project_to_random_angle(motion, dataset, mode):
    if mode == "uniform":
        hor_angle = np.random.uniform(-np.pi, np.pi)
    elif mode == "side":
        hor_angle = np.pi / 2
    elif mode == "hybrid":
        hor_angle = np.random.uniform(np.pi / 2 - np.pi / 4, np.pi / 2 + np.pi / 4)
    ver_angle = sample_vertical_angle(dataset)
    distance = sample_distance(dataset)
    projected_motion = perspective_projection(motion, [hor_angle], [ver_angle], distance)[0] * distance / get_visualization_scale(dataset)
    return projected_motion


from eval.test_evaluator import EvalSampler


class Evaluator:
    def __init__(self, args):
        self.encoder = EvalSampler(args)
        self.args = self.encoder.args

    def get_data_samples(self, split, **kwargs):
        return list(get_dataset_loader(self.args.dataset, split=split, data_size=self.args.eval_num_samples, batch_size=self.args.batch_size, **kwargs))

    def apply_on_data(self, func, split, **kwargs):
        return [func(input_motion, model_kwargs) for input_motion, model_kwargs in tqdm(self.get_data_samples(split, **kwargs))]

    def encode(self, motion, model_kwargs):
        return self.encoder.encode(motion, model_kwargs)[0]

    def encode_samples(self, iter):
        return [self.encode(motion, model_kwargs) for motion, model_kwargs in tqdm(iter)]

    @torch.no_grad()
    def get_model_samples(self, model_args, split="test"):
        # generate samples from the 2D diffusion model
        model_args = deepcopy(model_args)
        model_args.num_samples = model_args.batch_size
        model = Generator(model_args)

        def generate_motion(input_motion, model_kwargs):
            sample = model(save=False, visualize=False, model_kwargs=deepcopy(model_kwargs), progress=False)
            return model.transform(sample), model_kwargs

        return self.apply_on_data(generate_motion, split)

    @torch.no_grad()
    def get_mas_samples(self, mas_type_name, model_args, split="test"):
        model_args = deepcopy(model_args)
        model_args.num_samples = model_args.batch_size

        if mas_type_name in ["dreamfusion", "dreamfusion_annealed"]:
            model_args.num_views = 1

        model_3d: Sampler = MAS_TYPES[mas_type_name](model_args)

        def generate_motion(input_motion, model_kwargs):
            samples_3d = model_3d(save=False, visualize=False, model_kwargs=deepcopy(model_kwargs), progress=False)
            samples_2d = torch.stack([project_to_random_angle(sample_3d, self.args.dataset, self.args.angle_mode) for sample_3d in samples_3d]).permute(0, 2, 3, 1)
            return model_3d.transform(samples_2d), model_kwargs

        return self.apply_on_data(generate_motion, split)

    def get_3d_samples(self, samples_path, scale=1, flip=False, fps_ratio=1):
        motions = []
        file_names = os.listdir(samples_path)
        random.shuffle(file_names)  # Shuffling instead of random sampling can increase diversity and recall. Should not hurt FID and precision by much.
        for sample_file in tqdm(file_names[: self.args.eval_num_samples]):
            try:
                motion = np.load(os.path.join(samples_path, sample_file))
            except ValueError as e:
                print(f"Error loading {sample_file}: {e}")
                continue
            if 'bert' in samples_path:
                motion = convert_motionBert_skeleton(motion)
            if fps_ratio < 0:
                frames_to_interpolate = -fps_ratio
                motion = interpolate_motion(motion, frames_to_interpolate)
            else:
                motion = motion[::fps_ratio]  # fix fps
            motion[..., 1] *= -1 if flip else 1  # flip y axis
            motion *= scale  # scale
            motions.append(motion)

        lengths = [len(motion) for motion in motions]
        masks = [None for motion in motions]
        motions_batched = [zip(motions[i : i + self.args.batch_size], lengths[i : i + self.args.batch_size], masks[i : i + self.args.batch_size]) for i in range(0, len(motions), self.args.batch_size)]
        motions_batched = [collate(batch) for batch in motions_batched]
        motions_batched = [(project_to_random_angle(motion_batch.permute(0, 3, 1, 2), self.args.dataset, self.args.angle_mode).permute(0, 2, 3, 1), model_kwargs) for motion_batch, model_kwargs in motions_batched]
        motions_batched = [(self.encoder.transform(motion_batch.to(self.encoder.device)), model_kwargs) for motion_batch, model_kwargs in motions_batched]
        return motions_batched

    def get_multiple_samples(self, subjects, model_args=None):
        return {subject: self.get_samples(subject, model_args) for subject in subjects}

    def get_samples(self, subject, model_args=None):
        if "test_data" == subject:
            return self.get_data_samples("test")

        elif "motionBert" == subject:
            return self.get_3d_samples(MOTIONBERT_SAMPLES_PATH, scale=SCALE_FOR_SAVED_SAMPLES, flip=True, fps_ratio=2)  # get motionBert samples

        elif "ElePose" == subject:
            return self.get_3d_samples(ELEPOSE_SAMPLES_PATH, scale=1, flip=False, fps_ratio=-2)  # get ElePose samples

        elif "saved_samples" == subject:
            return self.get_3d_samples(self.args.saved_samples_path, scale=1.7, flip=True, fps_ratio=2)  # get saved samples

        elif "train_data" == subject:
            return self.get_data_samples("train")

        elif "model" == subject:
            return self.get_model_samples(model_args)
        
        elif "saved_samples" == subject:
            return self.get_3d_samples(self.args.saved_samples_path, scale=SCALE_FOR_SAVED_SAMPLES, flip=True, fps_ratio=1)

        elif subject in MAS_TYPES.keys():
            return self.get_mas_samples(subject, model_args)

        else:
            raise ValueError(f"Unknown subject {subject}")

    def visualize_samples(self, motions):
        if self.args.vis_subjects is None or len(self.args.vis_subjects) == 0 or self.args.vis_subjects[0] is None or self.args.vis_subjects[0] == "":
            return
        visualized = False
        for vis_subject in self.args.vis_subjects:
            if vis_subject in motions:
                for motion, model_kwargs in motions[vis_subject][: ceil(self.args.num_visualize_samples / self.args.batch_size)]:
                    visualized = True
                    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    name = f"{vis_subject}_sample_{date_time}"
                    print(f"visualizing to {name} a motion of length {motion.shape[1]}")
                    self.encoder.visualize(motion.to(self.encoder.device), name, model_kwargs=model_kwargs, normalized=True, num_samples=self.args.num_visualize_samples)
        assert visualized, f"No samples were visualized, expected one of {self.args.vis_subjects} to be in the samples"

    def evaluate_samples(self, samples, test_samples):
        # samples, test_samples: (B, J, 2, T)
        F = 30
        eps = 1e-6
        diameter_ratio_sum = 0.0
        num_frames = 0

        for motion_pair, test_motion_pair in zip(samples, test_samples):
            # motion, test_motion: (B, J, 2, T)
            # Take first F frames
            motion = motion_pair[0]
            test_motion = test_motion_pair[0].to(motion.device)
            mx = motion[:, :, 0, :F]
            my = motion[:, :, 1, :F]
            tx = test_motion[:, :, 0, :F]
            ty = test_motion[:, :, 1, :F]

            # Per-frame bbox: min/max over joints (axis=0)
            m_min_x = mx.min(axis=1).values; m_max_x = mx.max(axis=1).values
            m_min_y = my.min(axis=1).values; m_max_y = my.max(axis=1).values
            t_min_x = tx.min(axis=1).values; t_max_x = tx.max(axis=1).values
            t_min_y = ty.min(axis=1).values; t_max_y = ty.max(axis=1).values

            # Per-frame "diameter" (diagonal of bbox)
            m_diam = torch.sqrt((m_max_x - m_min_x) ** 2 + (m_max_y - m_min_y) ** 2)  # (B, F,)
            t_diam = torch.sqrt((t_max_x - t_min_x) ** 2 + (t_max_y - t_min_y) ** 2)  # (B, F,)
            # Accumulate ratios
            tensor = m_diam / (t_diam + eps)
            diameter_ratio_sum += torch.sum(tensor)
            num_frames += tensor.shape[0] * tensor.shape[1]

        diameter_ratio_mean = diameter_ratio_sum / max(num_frames, 1)
        print(f"Average dimater ratio: {diameter_ratio_mean}")
        encodings = convert_to_numpy(unbatch(self.encode_samples(samples)))
        test_encodings = convert_to_numpy(unbatch(self.encode_samples(test_samples)))

        all_metrics = {}
        if "fid" in self.args.metrics_names:
            all_metrics["fid"] = calculate_frechet_distance(encodings, test_encodings)

        if "diversity" in self.args.metrics_names:
            all_metrics["diversity"] = calculate_diversity(encodings, diversity_times=min(len(test_encodings) - 1, DIVERSITY_TIMES))

        if "precision" in self.args.metrics_names:
            all_metrics["precision"] = calculate_precision(encodings, test_encodings, k=PRECISION_AND_RECALL_K)

        if "recall" in self.args.metrics_names:
            all_metrics["recall"] = calculate_recall(encodings, test_encodings, k=PRECISION_AND_RECALL_K)

        return all_metrics

    def evaluate(self, model_args=None):
        all_samples = self.get_multiple_samples(self.args.subjects, model_args)
        self.visualize_samples(all_samples)

        test_samples = self.get_samples("test_data")
        all_metrics = {}

        self.visualize_samples(all_samples)
        for subject, samples in all_samples.items():
            metrics = self.evaluate_samples(samples, test_samples)
            all_metrics[subject] = metrics

        return all_metrics

    def evaluate_multiple_times(self, model_args=None, print=True, save=True):
        all_metrics = {subject: {metric_name: [] for metric_name in self.args.metrics_names} for subject in self.args.subjects}

        for i in range(self.args.num_eval_iterations):
            iter_metrics = self.evaluate(model_args)

            for subject, subject_iter_metrics in iter_metrics.items():
                for metric_name, metric in subject_iter_metrics.items():
                    all_metrics[subject][metric_name].append(metric)

        metrics_means, metrics_intervals = summarize_metrics(all_metrics, self.args)

        if print:
            print_table(metrics_means, metrics_intervals, self.args.subjects, self.args.metrics_names)

        if save:
            log_path = model_args.model_path.replace(".pt", ".pth").replace(".pth", f"_eval_{self.args.angle_mode}.txt")
            save_metrics(metrics_means, metrics_intervals, log_path, self.args.subjects, self.args.metrics_names)

        return metrics_means, metrics_intervals


def unbatch(batched_list):
    return sum([list(batch) for batch in batched_list], [])


def evaluate(evaluator_args, model_args):
    print(f"Evaluating [{model_args.model_path}]")
    evaluator = Evaluator(evaluator_args)
    return evaluator.evaluate_multiple_times(model_args)


def main():
    model_args, evaluator_args = evaluate_args()
    fixseed(evaluator_args.seed)
    evaluate(evaluator_args, model_args)


def summarize_metrics(all_metrics, evaluator_args):
    total_metrics = {subject: {metric_name: [] for metric_name in evaluator_args.metrics_names} for subject in evaluator_args.subjects}
    for subject, subject_all_metrics in all_metrics.items():
        for metric_name, subject_metric in subject_all_metrics.items():
            total_metrics[subject][metric_name].append(subject_metric)

    total_means = {subject: {metric_name: np.mean(metrics) for metric_name, metrics in subject_all_metrics.items()} for subject, subject_all_metrics in total_metrics.items()}
    total_stds = {subject: {metric_name: np.std(metrics) for metric_name, metrics in subject_all_metrics.items()} for subject, subject_all_metrics in total_metrics.items()}
    conf_intervals = {subject: {metric_name: 1.96 * std / np.sqrt(evaluator_args.num_eval_iterations) for metric_name, std in subject_stds.items()} for subject, subject_stds in total_stds.items()}

    return total_means, conf_intervals


def format_float(f):
    if f"{f:.2f}" == "0.00":
        return f"{f:.2e}"
    return f"{f:.2f}"


def print_table(total_means, conf_intervals, subjects, metrics_names):
    print(" " * 24 + "".join([f"{metric_name: <20}" for metric_name in (metrics_names)]))
    for subject in subjects:
        line = f"{subject: <24}"
        for metric_name in metrics_names:
            metrics = f"{format_float(total_means[subject][metric_name])}±{format_float(conf_intervals[subject][metric_name])}"
            line += f"{metrics: <20}"
        print(line)


def save_metrics(metrics_means, metrics_intervals, log_file_path, subjects, metrics_names):
    sys.stdout = open(log_file_path, "w")
    print_table(metrics_means, metrics_intervals, subjects, metrics_names)
    sys.stdout = sys.__stdout__
    print(f"Saved eval log to [{log_file_path}]")


if __name__ == "__main__":
    main()
