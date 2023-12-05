# See: https://github.com/zhulf0804/PointPillars (unofficial) and:
# https://github.com/nutonomy/second.pytorch (official).

import numpy as np
import torch

from numba import float32, int32, njit
from torch import LongTensor, nn, Tensor

# See: https://github.com/nutonomy/second.pytorch/blob/449c7c0d081eaad44f08159f64af26d2a59f1f4c/second/pytorch/models/voxelnet.py#L353.
EPS = 1e-3
MOMENTUM = 0.01


# See Section 2.1.
@njit(
    (
        float32[:, :],
        int32,
        int32,
        int32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
        float32,
    )
)
def pillarize(
    points,
    max_pillars,
    max_points,
    side_cells,
    min_x,
    max_x,
    min_y,
    max_y,
    min_z,
    max_z,
    mid_x,
    mid_y,
    mid_z,
    half_x_length,
    half_y_length,
    half_z_length,
):
    n_points = points.shape[0]
    idxs = np.random.choice(n_points, n_points, replace=False)
    points = points[idxs]

    pillars_buffer = np.zeros((max_pillars, max_points, 6), dtype="float32")
    pillar_pixels = np.full((max_pillars, 2), -1, dtype="int32")
    pillar2idx = np.full((side_cells, side_cells), -1, dtype="int32")
    point_counts = np.zeros(max_pillars, dtype="int32")
    pillar_avgs = np.zeros((max_pillars, 3), dtype="float32")

    max_idx = side_cells - 1
    pillar_count = 0
    for x, y, z in points:
        # Skip points that are not inside the enclosing prism.
        in_x = min_x < x < max_x
        in_y = min_y < y < max_y
        in_z = min_z < z < max_z
        if not (in_x & in_y & in_z):
            continue

        # Shift so prism is centered at (0, 0, 0).
        x -= mid_x
        y -= mid_y
        z -= mid_z
        # Scale so that values *inside* prism are in (-1, 1).
        x /= half_x_length
        y /= half_y_length
        z /= half_z_length

        # Transform coordinates from (-1, 1) to (0, 1).
        row_prop = 1 - (x + 1) / 2
        col_prop = (y + 1) / 2
        # Convert values to grid indexes while handling numerical precision issues.
        row = min(int32(row_prop * side_cells), max_idx)
        col = min(int32(col_prop * side_cells), max_idx)
        pillar = (row, col)
        idx = pillar2idx[pillar]
        if idx == -1:
            if pillar_count == max_pillars:
                continue

            idx = pillar_count
            pillar2idx[pillar] = idx
            pillar_pixels[idx] = (row, col)
            pillar_count += 1

        point_count = point_counts[idx]
        if point_count == max_points:
            continue

        pillars_buffer[idx, point_count, 0] = x
        pillars_buffer[idx, point_count, 1] = y
        pillars_buffer[idx, point_count, 2] = z

        point_counts[idx] += 1
        # See: https://stackoverflow.com/a/61377432/1316276.
        a = 1 / point_counts[idx]
        b = 1 - a
        pillar_avgs[idx, 0] = a * x + b * pillar_avgs[idx, 0]
        pillar_avgs[idx, 1] = a * y + b * pillar_avgs[idx, 1]
        pillar_avgs[idx, 2] = a * z + b * pillar_avgs[idx, 2]

    return (pillars_buffer, pillar_pixels, pillar_avgs)


def get_prepare_pillars_function(
    max_pillars, max_points, side_cells, x_range, y_range, z_range
):
    max_idx = side_cells - 1

    # Precompute values that will be used in pillarize function.
    (min_x, max_x) = x_range
    (min_y, max_y) = y_range
    (min_z, max_z) = z_range

    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2
    mid_z = (min_z + max_z) / 2

    half_x_length = (max_x - min_x) / 2
    half_y_length = (max_y - min_y) / 2
    half_z_length = (max_z - min_z) / 2

    def prepare_pillars(points, labels=None):
        pillar_pieces = pillarize(
            points.astype("float32"),
            max_pillars,
            max_points,
            side_cells,
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
            mid_x,
            mid_y,
            mid_z,
            half_x_length,
            half_y_length,
            half_z_length,
        )

        tgt = None
        if labels is not None:
            if len(labels.shape) == 1:
                labels = labels[None]

            tgt = np.zeros((side_cells, side_cells), dtype="float32")
            for label in labels:
                center_x = label[0]
                center_y = label[1]
                in_x = min_x < center_x < max_x
                in_y = min_y < center_y < max_y
                if not (in_x and in_y):
                    continue

                # Convert coordinates to grid indexes using the transformation from the
                # pillarize function.
                center_x -= mid_x
                center_y -= mid_y
                center_x /= half_x_length
                center_y /= half_y_length
                row_prop = 1 - (center_x + 1) / 2
                col_prop = (center_y + 1) / 2
                row = min(int32(row_prop * side_cells), max_idx)
                col = min(int32(col_prop * side_cells), max_idx)

                tgt[row, col] = 1

        return (pillar_pieces, tgt)

    return prepare_pillars


class PointNetLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Conv2d(in_feats, out_feats, 1)
        self.norm = nn.BatchNorm2d(out_feats)
        self.relu = nn.ReLU()

    def forward(self, points):
        pn_feats = self.relu(self.norm(self.linear(points)))
        return pn_feats


class PointNet(nn.Module):
    def __init__(self, in_feats, mlp_layers, position_encodings):
        super().__init__()
        in_feats = (1 + 2 * position_encodings) * in_feats
        self.L = position_encodings
        pointnet = []
        for out_feats in mlp_layers:
            pointnet.append(PointNetLayer(in_feats, out_feats))
            in_feats = out_feats

        self.pointnet = nn.Sequential(*pointnet)

    def forward(self, pillar_buffers):
        pillar_buffers_encoded = [pillar_buffers]
        for l_pos in range(self.L):
            pillar_buffers_encoded.append(
                torch.sin(2**l_pos * torch.pi * pillar_buffers)
            )
            pillar_buffers_encoded.append(
                torch.cos(2**l_pos * torch.pi * pillar_buffers)
            )

        pillar_buffers_encoded = torch.cat(pillar_buffers_encoded, dim=1)

        pn_feats = self.pointnet(pillar_buffers_encoded)
        # See: https://github.com/pytorch/pytorch/issues/47163#issuecomment-757122069.
        return pn_feats.max(3)[0].permute(0, 2, 1).contiguous()


class Block(nn.Module):
    def __init__(self, in_channels, stride, layers, out_channels):
        super().__init__()
        block = [
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=stride,
                    bias=False,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels, eps=EPS, momentum=MOMENTUM),
                nn.ReLU(inplace=True),
            )
        ]
        for _ in range(layers - 1):
            block.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, bias=False, padding=1),
                    nn.BatchNorm2d(out_channels, eps=EPS, momentum=MOMENTUM),
                    nn.ReLU(inplace=True),
                )
            )

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_stride):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                upsample_stride,
                stride=upsample_stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=EPS, momentum=MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(x)


class Backbone(nn.Module):
    def __init__(self, in_channels, block_layers):
        super().__init__()
        blocks = []
        ups = []
        up_stride = 1
        for stride, layers, out_channels, up_channels in block_layers:
            blocks.append(Block(in_channels, stride, layers, out_channels))
            # See Section 3.1 and Figure 2.
            up_stride *= stride
            ups.append(Up(out_channels, up_channels, up_stride))
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.ups = nn.ModuleList(ups)

    def forward(self, x):
        up_xs = []
        for block, up in zip(self.blocks, self.ups):
            x = block(x)
            up_xs.append(up(x))

        return torch.cat(up_xs, dim=1)


class Predictor(nn.Module):
    def __init__(self, backbone, head, side_cells):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.side_cells = side_cells

    def forward(self, pn_feats, pillar_pixels):
        (N, P, C) = pn_feats.shape
        pseudo_images = torch.zeros(N, C, self.side_cells, self.side_cells).to(pn_feats)
        batch_idxs = torch.arange(N).repeat_interleave(P).to(pillar_pixels)
        rows = pillar_pixels[..., 0].flatten()
        cols = pillar_pixels[..., 1].flatten()
        # -1 rows are empty points.
        mask = rows != -1
        batch_idxs = batch_idxs[mask]
        rows = rows[mask]
        cols = cols[mask]
        pseudo_images[batch_idxs, :, rows, cols] = pn_feats.view(-1, C)[mask]
        preds = self.head(self.backbone(pseudo_images)).squeeze(1)

        return preds


class PointPillars(nn.Module):
    def __init__(
        self,
        position_encodings,
        mlp_layers,
        block_layers,
        side_cells,
        x_range,
        y_range,
        z_range,
        max_pillars,
        max_points,
    ):
        super().__init__()
        # Three for raw point coordinates plus three for difference from mean point in
        # cell.
        in_feats = 3 + 3
        self.pointnet = PointNet(in_feats, mlp_layers, position_encodings)
        backbone = Backbone(mlp_layers[-1], block_layers)
        in_channels = sum([block_layer[-1] for block_layer in block_layers])
        # Presence/absence in grid cell.
        head = nn.Conv2d(in_channels, 1, 1)
        # Predictor takes in a pseudo image, extracts features using the backbone, and
        # then predicts a presence/absence grid using the head.
        self.predictor = Predictor(backbone, head, side_cells)

        # Set up the prepare_pillars function.
        self.max_pillars = max_pillars
        self.max_points = max_points
        self.prepare_pillars = get_prepare_pillars_function(
            max_pillars, max_points, side_cells, x_range, y_range, z_range
        )

        # Stuff for predicting detections.
        self.register_buffer("prob_idxs", torch.arange(side_cells**2).long())
        self.register_buffer("side_cells", torch.LongTensor([side_cells]))
        self.register_buffer("offset", (1 / self.side_cells) / 2)
        self.register_buffer("x_scale", torch.Tensor([x_range[1] - x_range[0]]))
        self.register_buffer("x_shift", torch.Tensor([x_range[0]]))
        self.register_buffer("y_scale", torch.Tensor([y_range[1] - y_range[0]]))
        self.register_buffer("y_shift", torch.Tensor([y_range[0]]))
        self.cell_length = (x_range[1] - x_range[0]) / side_cells
        self.cell_width = (y_range[1] - y_range[0]) / side_cells

    def __str__(self):
        desc = f"PointPillars\n"
        desc += f"max_pillars: {self.max_pillars}\n"
        desc += f"max_points: {self.max_points}\n"
        desc += f"side_cells: {self.side_cells[0].item()}\n"
        desc += f"cell_length: {self.cell_length}\n"
        desc += f"cell_width: {self.cell_width}"
        return desc

    def get_detections(self, points, device, min_logit):
        pillar_pieces = self.prepare_pillars(points)[0]
        pillar_buffers = Tensor(pillar_pieces[0])[None].to(device)
        pillar_pixels = LongTensor(pillar_pieces[1])[None].to(device)
        pillar_avgs = Tensor(pillar_pieces[2])[None].to(device)

        preds = self(pillar_buffers, pillar_avgs, pillar_pixels)

        keep_idxs = self.prob_idxs[preds[0].flatten() > min_logit]
        rows = keep_idxs // self.side_cells
        cols = keep_idxs % self.side_cells

        # Convert rows and columns into world coordinates.
        xs = self.x_scale * (1 - rows / self.side_cells - self.offset) + self.x_shift
        ys = self.y_scale * (cols / self.side_cells + self.offset) + self.y_shift
        centers = torch.stack([xs, ys]).permute(1, 0)

        return centers

    def forward(self, pillar_buffers, pillar_avgs, pillar_pixels):
        P = pillar_buffers.shape[2]
        pillar_avgs = pillar_avgs.unsqueeze(2).expand(-1, -1, P, -1)
        pillar_buffers[..., 3:] = pillar_buffers[..., :3] - pillar_avgs
        # These are empty points, so we zero out their center offset.
        mask = (pillar_buffers[..., :3].sum(-1) == 0).unsqueeze(3)
        pillar_buffers[mask.squeeze(-1)] = 0
        pillar_buffers = pillar_buffers.permute(0, 3, 1, 2).contiguous()

        # See Section 2.1. Each pillar buffer is processed independently by a PointNet.
        pn_feats = self.pointnet(pillar_buffers)

        # See Section 2.2 and Section 2.3.
        preds = self.predictor(pn_feats, pillar_pixels)

        return preds
