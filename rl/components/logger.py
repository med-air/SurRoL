import csv
import datetime
import inspect
import logging
import os
from collections import defaultdict

import colorlog
import numpy as np
import torch
import wandb
from termcolor import colored

from utils.general_utils import flatten_dict, np2obj, prefix_dict
from utils.vis_utils import add_captions_to_seq

#----------------------Termnial Logger----------------------
formatter = colorlog.ColoredFormatter(
    "%(asctime_log_color)s[%(asctime)s]%(name_log_color)s[%(name)s]%(levelname_log_color)s[%(levelname)s] - %(message_log_color)s%(message)s",
    datefmt=None,
    reset=True,
    secondary_log_colors={
        'asctime': {
            'INFO': 'cyan',
            'ERROR': 'cyan'
        },
        'name': {
            'INFO': 'blue',
            'ERROR': 'blue'
        },
        'levelname': {
            'INFO': 'green',
            'ERROR': 'red'
        },
        'message': {
            'INFO': 'white',
            'ERROR': 'red'
        }
    },
    style="%",
)

logger = colorlog.getLogger("dex")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    ch = colorlog.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


#----------------------WandB Logger----------------------
class WandBLogger:
    """Logs to WandB."""
    N_LOGGED_SAMPLES = 50    # how many examples should be logged in each logging step

    def __init__(self, exp_name, project_name, entity, path, conf, exclude=None):
        """
        :param exp_name: full name of experiment in WandB
        :param project_name: name of overall project
        :param entity: name of head entity in WandB that hosts the project
        :param path: path to which WandB log-files will be written
        :param conf: hyperparam config that will get logged to WandB
        :param exclude: (optional) list of (flattened) hyperparam names that should not get logged
        """
        if exclude is None: exclude = []
        flat_config = flatten_dict(conf)
        filtered_config = {k: v for k, v in flat_config.items() if (k not in exclude and not inspect.isclass(v))}
        
        # clear dir
        # save_dir = path / 'wandb'
        # save_dir.mkdir(exist_ok=True)
        # shutil.rmtree(f"{save_dir}/")

        logger.info("Init wandb")
        wandb.init(
            resume=exp_name,
            project=project_name,
            config=filtered_config,
            dir=path,
            entity=entity,
            notes=conf.notes if 'notes' in conf else ''
        )

    def log_scalar_dict(self, d, prefix='', step=None):
        """Logs all entries from a dict of scalars. Optionally can prefix all keys in dict before logging."""
        if prefix: d = prefix_dict(d, prefix + '_')
        wandb.log(d) if step is None else wandb.log(d, step=step)

    def log_videos(self, vids, name, step=None):
        """Logs videos to WandB in mp4 format.
        Assumes list of numpy arrays as input with [time, channels, height, width]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        if vids[0].max() <= 1.0: vids = [np.asarray(vid * 255.0, dtype=np.uint8) for vid in vids]
        # TODO(karl) expose the FPS as a parameter
        log_dict = {name: [wandb.Video(vid, fps=10, format="mp4") for vid in vids]}
        wandb.log(log_dict) if step is None else wandb.log(log_dict, step=step)

    def log_plot(self, fig, name, step=None):
        """Logs matplotlib graph to WandB.
        fig is a matplotlib figure handle."""
        img = wandb.Image(fig)
        wandb.log({name: img}) if step is None else wandb.log({name: img}, step=step)

    def log_outputs(self, logging_stats, rollout_storage, log_images, step, is_train=False, log_videos=True, log_video_caption=False):
        """Visualizes/logs all training outputs."""
        self.log_scalar_dict(logging_stats, prefix='train' if is_train else 'eval', step=step)

        if log_images:
            assert rollout_storage is not None      # need rollout data for image logging
            # log rollout videos with info captions
            if 'image' in rollout_storage and log_videos:
                if log_video_caption:
                    vids = [np.stack(add_captions_to_seq(rollout.image, np2obj(rollout.info))).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-self.n_logged_samples:]]
                else:
                    vids = [np.stack(rollout.image).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-self.n_logged_samples:]]
                self.log_videos(vids, name="rollouts", step=step)

    @property
    def n_logged_samples(self):
        # TODO(karl) put this functionality in a base logger class + give it default parameters and config
        return self.N_LOGGED_SAMPLES
    

#----------------------CSV Logger----------------------
COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'), ('episode_sr', 'SR', 'float'),
                       ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                       ('total_time', 'T', 'time'), ('ETA', 'ETA', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        if(os.path.exists(csv_file_name) and os.path.isfile(csv_file_name)):
            os.remove(csv_file_name)

        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # print(row)
                # if float(row['episode']) >= data['episode']:
                #     break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'blue' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        logger.info(' | '.join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._train_mg = MetersGroup(log_dir / 'train.csv',
                                     formating=COMMON_TRAIN_FORMAT)
        self._eval_mg = MetersGroup(log_dir / 'eval.csv',
                                    formating=COMMON_EVAL_FORMAT)
        self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty is None or ty == 'train':
            self._train_mg.dump(step, 'train')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)