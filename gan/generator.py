import torch
from torch import nn
from .temp_network import TemporalNetwork
from .bar_generator import BarGenerator

class MuseGenerator(nn.Module):
    """
    A generator model for creating musical compositions.

    Attributes:
    - z_dimension (int): Dimensionality of the input noise vector. Defaults to 32.
    - hid_channels (int): Number of hidden channels in the temporal and bar generator networks. Defaults to 1024.
    - hid_features (int): Number of hidden features in the bar generator network. Defaults to 1024.
    - out_channels (int): Number of output channels, typically corresponding to the generated output's depth. Defaults to 1.
    - n_tracks (int): Number of individual tracks to generate within each composition. Defaults to 4.
    - n_bars (int): Number of bars per track in the composition. Defaults to 2.
    - n_steps_per_bar (int): Number of steps per bar. Defaults to 16.
    - n_pitches (int): Number of pitches per step. Defaults to 84.
    """

    def __init__(self, z_dimension=32, hid_channels=1024, hid_features=1024, out_channels=1, n_tracks=4, n_bars=2, n_steps_per_bar=16, n_pitches=84):
        super(MuseGenerator, self).__init__()
        self.n_tracks = n_tracks
        self.n_bars = n_bars
        self.n_steps_per_bar = n_steps_per_bar
        self.n_pitches = n_pitches

        # Initialize the networks for chords and melodies generation
        self.chords_network = TemporalNetwork(z_dimension, hid_channels, n_bars=n_bars)
        self.melody_networks = nn.ModuleDict({
            f"melodygen_{n}": TemporalNetwork(z_dimension, hid_channels, n_bars=n_bars)
            for n in range(n_tracks)
        })

        # Initialize bar generators for each track
        self.bar_generators = nn.ModuleDict({
            f"bargen_{n}": BarGenerator(z_dimension, hid_features, hid_channels // 2, out_channels, n_steps_per_bar, n_pitches)
            for n in range(n_tracks)
        })

    def forward(self, chords, style, melody, groove):
        """
        Forward pass to generate a musical composition.

        Parameters:
        - chords (Tensor): Tensor representing chord progressions.
        - style (Tensor): Tensor representing stylistic elements.
        - melody (Tensor): Tensor representing melodic elements for each track.
        - groove (Tensor): Tensor representing rhythmic elements for each track.

        Returns:
        - Tensor: The generated musical composition.
        """
        chord_outs = self.chords_network(chords)
        bar_outs = []

        for bar in range(self.n_bars):
            track_outs = []
            chord_out = chord_outs[:, :, bar]
            style_out = style
            for track in range(self.n_tracks):
                melody_in = melody[:, track, :]
                melody_out = self.melody_networks[f"melodygen_{track}"](melody_in)[:, :, bar]
                groove_out = groove[:, track, :]
                z = torch.cat([chord_out, style_out, melody_out, groove_out], dim=1)
                track_outs.append(self.bar_generators[f"bargen_{track}"](z))
            track_out = torch.cat(track_outs, dim=1)
            bar_outs.append(track_out)
        out = torch.cat(bar_outs, dim=2)
        return out
