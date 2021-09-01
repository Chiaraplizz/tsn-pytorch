import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='image_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.arch= 'i3d'
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            if not os.path.exists(os.path.join(directory, self.image_tmpl.format(idx))):
                print("Not found: ", os.path.join(directory, self.image_tmpl.format(idx)))
                return [Image.open('/home/chiarap/dataset/ucf101/RGB-feature/v_GolfSwing_g22_c01/img_00200.jpg').convert('RGB')]
            else:
                return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        print(len(self.video_list))
    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.arch == 'i3d':
            n_frames = record.num_frames
            n_frames_per_segment = 16

            stride = 1
            idces_start = 0 if n_frames <= n_frames_per_segment * stride \
                else \
                np.random.randint(low=0, high=n_frames - (n_frames_per_segment * stride))
            offsets = [x + 1 for x in range(idces_start, idces_start + n_frames_per_segment * stride, stride)]
        else:
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_val_indices(self, record):
        if self.arch == 'i3d':
            stride = 1
            n_frames_per_segment = 16

            # calculate the distance between one segment and the following
            # if < 0, then it means that the two frames overlap proportionally to space_between_segs
            if record.num_frames > 16*6:
                center_frames = np.linspace(0,
                                        record.num_frames,
                                        4 + 2,
                                        dtype=np.int32)[1:-1]
            else:
                center_frames = np.linspace(0,
                                            record.num_frames,
                                            6 + 2,
                                            dtype=np.int32)[2:-2]

            # indices = [0, ..., n_frames_per_segment-1, step, ..., step + n_frames_per_segment-1, ...]
            # repeated exactly n_frames_
            # each 'frame' is the center of each segment
            offsets = [x + 1
                       for center in center_frames
                       for x in range(center - n_frames_per_segment // 2 * stride,  # start of the segment
                                      center + n_frames_per_segment // 2 * stride,  # end of the segment
                                      stride)  # step of the sampling
                       ]
        else:

            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        if self.arch == 'i3d':
            stride = 1
            n_frames_per_segment = 16

            # calculate the distance between one segment and the following
            # if < 0, then it means that the two frames overlap proportionally to space_between_segs
            center_frames = np.linspace(record.start_frame,
                                        record.end_frame,
                                        4 + 2,
                                        dtype=np.int32)[1:-1]

            # indices = [0, ..., n_frames_per_segment-1, step, ..., step + n_frames_per_segment-1, ...]
            # repeated exactly n_frames_
            # each 'frame' is the center of each segment
            offsets = [x + 1 - record.start_frame
                       for center in center_frames
                       for x in range(center - n_frames_per_segment // 2 * stride,  # start of the segment
                                      center + n_frames_per_segment // 2 * stride,  # end of the segment
                                      stride)  # step of the sampling
                       ]
        else:

            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
