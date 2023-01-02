"""Stereo matching."""
import numpy as np
from scipy.signal import convolve2d


class Solution:
    def __init__(self):
        pass

    @staticmethod
    def ssd_distance(left_image: np.ndarray,
                     right_image: np.ndarray,
                     win_size: int,
                     dsp_range: int) -> np.ndarray:
        """Compute the SSDD distances tensor.

        Args:
            left_image: Left image of shape: HxWx3, and type np.double64.
            right_image: Right image of shape: HxWx3, and type np.double64.
            win_size: Window size odd integer.
            dsp_range: Half of the disparity range. The actual range is
            -dsp_range, -dsp_range + 1, ..., 0, 1, ..., dsp_range.

        Returns:
            A tensor of the sum of squared differences for every pixel in a
            window of size win_size X win_size, for the 2*dsp_range + 1
            possible disparity values. The tensor shape should be:
            HxWx(2*dsp_range+1).
        """
        num_of_rows, num_of_cols = left_image.shape[0], left_image.shape[1]
        disparity_values = range(-dsp_range, dsp_range + 1)
        ssdd_tensor = np.zeros((num_of_rows,
                                num_of_cols,
                                len(disparity_values)))

        half_win = win_size // 2
        pad_x_right = dsp_range + half_win
        left_padded = np.pad(left_image, ((half_win, half_win), (half_win, half_win), (0, 0)))
        right_padded = np.pad(right_image, ((half_win, half_win), (pad_x_right, pad_x_right), (0, 0)))

        kernel = np.ones((win_size, win_size))
        overlap = dsp_range + num_of_cols + half_win * 2
        for d_index, d in enumerate(disparity_values):
            diff = left_padded - right_padded[:, d + dsp_range:d + overlap]
            squared_diff = np.sum(diff ** 2, axis=2)
            sum_squared_diff = convolve2d(squared_diff, kernel, mode='valid')
            ssdd_tensor[:, :, d_index] = sum_squared_diff

        ssdd_tensor -= ssdd_tensor.min()
        ssdd_tensor /= ssdd_tensor.max()
        ssdd_tensor *= 255.0
        return ssdd_tensor

    @staticmethod
    def naive_labeling(ssdd_tensor: np.ndarray) -> np.ndarray:
        """Estimate a naive depth estimation from the SSDD tensor.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.

        Evaluate the labels in a naive approach. Each value in the
        result tensor should contain the disparity matching minimal ssd (sum of
        squared difference).

        Returns:
            Naive labels HxW matrix.
        """
        label_no_smooth = np.argmin(ssdd_tensor, axis=2)
        return label_no_smooth

    @staticmethod
    def dp_grade_slice(c_slice: np.ndarray, p1: float, p2: float) -> np.ndarray:
        """Calculate the scores matrix for slice c_slice.

        Calculate the scores slice which for each column and disparity value
        states the score of the best route. The scores slice is of shape:
        (2*dsp_range + 1)xW.

        Args:
            c_slice: A slice of the ssdd tensor.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Scores slice which for each column and disparity value states the
            score of the best route.
        """
        num_labels, num_of_cols = c_slice.shape[0], c_slice.shape[1]
        l_slice = np.zeros((num_labels, num_of_cols))
        l_slice[:, 0] = c_slice[:, 0]

        xx, yy = np.meshgrid(np.arange(num_labels), np.arange(num_labels))
        for col in range(1, num_of_cols):
            l = l_slice[:, col - 1]
            M = np.tile(l, [num_labels, 1])
            M[np.abs(yy - xx) == 1] += p1
            M[np.abs(yy - xx) >= 2] += p2

            l_slice[:, col] = c_slice[:, col] + np.min(M, axis=1) - np.min(l)
        return l_slice

    def dp_labeling(self,
                    ssdd_tensor: np.ndarray,
                    p1: float,
                    p2: float) -> np.ndarray:
        """Estimate a depth map using Dynamic Programming.

        (1) Call dp_grade_slice on each row slice of the ssdd tensor.
        (2) Store each slice in a corresponding l tensor (of shape as ssdd).
        (3) Finally, for each pixel in l (along each row and column), choose
        the best disparity value. That is the disparity value which
        corresponds to the lowest l value in that pixel.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.
        Returns:
            Dynamic Programming depth estimation matrix of shape HxW.
        """
        l = np.zeros_like(ssdd_tensor)
        for row in range(ssdd_tensor.shape[0]):
            row_labels = self.dp_grade_slice(ssdd_tensor[row, :, :].T, p1, p2)
            l[row, :] = row_labels.T
        return self.naive_labeling(l)

    def extract_slice_indices_generator(self, ssdd_tensor: np.ndarray, direction: int) -> np.ndarray:
        """
         extracts slices from the ssdd according to a direction which it
         receives as an input.
        Args:
            ssdd_tensor: A tensor of the sum of squared differences for every
            pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            direction: an integer in 1, ..., 8, representing the direction to extract slices along.
        Returns:
            A slices from the ssdd_tensor according to the input direction
        """
        num_of_rows, num_of_cols = ssdd_tensor.shape[:2]
        if direction % 2 == 0:  # diagonal
            xv, yv = np.meshgrid(np.arange(num_of_rows), np.arange(num_of_cols), indexing='ij')

            if direction == 4:
                yv = np.fliplr(yv)

            elif direction == 6:
                xv = np.flipud(xv)
                yv = np.fliplr(yv)

            elif direction == 8:
                xv = np.flipud(xv)

            num_iters = num_of_cols - 1
            iter_id = -num_of_rows + 1
            while iter_id < num_iters:
                yield np.diag(xv, iter_id), np.diag(yv, iter_id)
                iter_id += 1
        else:
            iter_id = 0
            if direction == 1:
                xv, yv = np.meshgrid(np.arange(num_of_rows), np.arange(num_of_cols), indexing='ij')
                num_iters = num_of_rows
            elif direction == 5:
                xv, yv = np.meshgrid(np.arange(num_of_rows), np.arange(num_of_cols), indexing='ij')
                yv = np.fliplr(yv)
                num_iters = num_of_rows
            elif direction == 3:
                xv, yv = np.meshgrid(np.arange(num_of_rows), np.arange(num_of_cols), indexing='xy')
                num_iters = num_of_cols
            else:
                xv, yv = np.meshgrid(np.arange(num_of_rows), np.arange(num_of_cols), indexing='xy')
                xv = np.fliplr(xv)
                num_iters = num_of_cols
            while iter_id < num_iters:
                yield xv[iter_id, :], yv[iter_id, :]
                iter_id += 1

    def dp_labeling_per_direction(self,
                                  ssdd_tensor: np.ndarray,
                                  p1: float,
                                  p2: float) -> dict:
        """Return a dictionary of directions to a Dynamic Programming
        etimation of depth.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Dictionary int->np.ndarray which maps each direction to the
            corresponding dynamic programming estimation of depth based on
            that direction.
        """
        num_of_directions = 8
        direction_to_slice = {}
        for direction in range(1, num_of_directions + 1):
            l = np.zeros_like(ssdd_tensor)
            for slice_ids in self.extract_slice_indices_generator(ssdd_tensor, direction):
                l[slice_ids] = self.dp_grade_slice(ssdd_tensor[slice_ids].T, p1, p2).T
            direction_to_slice[direction] = self.naive_labeling(l)
        return direction_to_slice

    def sgm_labeling(self, ssdd_tensor: np.ndarray, p1: float, p2: float):
        """Estimate the depth map according to the SGM algorithm.

        For each direction in 1, ..., 8, calculate scores tensors
        according to dp_grade_slice and the method which allows you to
        extract slices along each direction.

        You may use helper methods (functions) that you write on your own.
        We found `np.diagonal` to be very helpful to extract diagonal slices.
        `np.unravel_index` might be helpful if you're thinking in MATLAB
        notations: it's the ind2sub equivalent.

        Args:
            ssdd_tensor: A tensor of the sum of squared differences for
            every pixel in a window of size win_size X win_size, for the
            2*dsp_range + 1 possible disparity values.
            p1: penalty for taking disparity value with 1 offset.
            p2: penalty for taking disparity value more than 2 offset.

        Returns:
            Semi-Global Mapping depth estimation matrix of shape HxW.
        """
        num_of_directions = 8
        ssdd_shape = list(ssdd_tensor.shape)
        ssdd_shape.insert(0, num_of_directions)
        ssdd_per_direction = np.zeros(ssdd_shape)
        for direction in range(1, num_of_directions + 1):
            l = np.zeros_like(ssdd_tensor)
            for slice_ids in self.extract_slice_indices_generator(ssdd_tensor, direction):
                l[slice_ids] = self.dp_grade_slice(ssdd_tensor[slice_ids].T, p1, p2).T
            ssdd_per_direction[direction - 1] = l
        l = np.mean(ssdd_per_direction, axis=0)
        return self.naive_labeling(l)
