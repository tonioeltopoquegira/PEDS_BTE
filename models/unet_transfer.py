from flax import nnx
import jax
import jax.numpy as jnp

from flax import nnx
import jax.numpy as jnp
import jax

class unet(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        super().__init__()

        self.encoder_filters = [1, 16]  # Filters in downsampling path
        self.decoder_filters = [16, 1]  # Filters in upsampling path
        self.bottleneck_filters = 64     # Bottleneck filters

        # Encoder layers
        self.encoder = [
            nnx.Conv(in_features=i, out_features=o, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                     kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
            for i,o in zip(self.encoder_filters[:-1], self.encoder_filters[1:])
        ]
        
        self.bottleneck = nnx.Conv(in_features=32, out_features=32, kernel_size=(3, 3),
                                   strides=(1, 1), padding="SAME",
                                   kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
        # Decoder layers
        self.decoder = [
            nnx.ConvTranspose(in_features=i, out_features=o, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                              kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
            for i,o in zip(self.decoder_filters[:-1], self.decoder_filters[1:])
        ]
        self.output_layer = nnx.Conv(in_features=1, out_features=1, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                                     kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)

        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

    @nnx.jit
    def __call__(self, x):
        # Input shape: (batch_size, 20, 20)

        # Downsampling path
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            x = nnx.tanh(x)  # Use tanh activation
            encoder_outputs.append(x)
            x = jax.lax.reduce_window(x, init_value=0.0, computation=jax.lax.add, 
                                      window_dimensions=(2, 2), window_strides=(2, 2), padding='SAME')  # Max pooling

        # Bottleneck
        x = self.bottleneck(x)
        x = nnx.tanh(x)
        x = self.dropout(x)

        # Upsampling path
        for layer, skip_connection in zip(self.decoder, reversed(encoder_outputs)):
            x = layer(x)
            x = nnx.tanh(x)
            x = jnp.concatenate([x, skip_connection], axis=-1)  # Skip connection

        # Output layer
        x = self.output_layer(x)
        x = nnx.tanh(x)
        x = 150.0 * x  # Scale the output
        x = jnp.minimum(200.0, x)
        x = jnp.maximum(-50.0, x)

        # Output shape: (batch_size, 20, 20)
        return x