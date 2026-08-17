import tensorflow as tf
import json
import zipfile
import h5py


MODEL_FILE = "crowd_counting_unet.keras"
WEIGHTS_FILE = "crowd_counting_unet.weights.h5"
OUTPUT_FILE = "crowd_counting_unet_clean.keras"


print("TensorFlow:")
print(tf.__version__)


# ==========================================================
# 1. Extract model architecture from .keras WITHOUT loading weights
# ==========================================================

print("\nExtracting config...")

with zipfile.ZipFile(MODEL_FILE, "r") as z:
    config = json.loads(
        z.read("config.json")
    )


# Remove training information
config.pop(
    "compile_config",
    None
)


print("Creating architecture...")


model = tf.keras.saving.deserialize_keras_object(
    config
)


print("Architecture created")

print(
    "Input:",
    model.input_shape
)

print(
    "Output:",
    model.output_shape
)


# ==========================================================
# 2. Get convolution layers from model
# ==========================================================

conv_layers = [
    layer
    for layer in model.layers
    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.Conv2DTranspose
        )
    )
]


print(
    "\nConv layers in model:",
    len(conv_layers)
)


for i, layer in enumerate(conv_layers):
    print(
        i,
        layer.name
    )


# ==========================================================
# 3. Read convolution weights from H5
# ==========================================================

print("\nReading weights...")


with h5py.File(
    WEIGHTS_FILE,
    "r"
) as f:

    all_groups = list(
        f["_layer_checkpoint_dependencies"].keys()
    )


# Only convolution layers contain kernel/bias
conv_weight_groups = [
    x
    for x in all_groups
    if x.startswith("conv2d")
]


def conv_number(name):

    if name == "conv2d":
        return 0

    return int(
        name.replace(
            "conv2d_",
            ""
        )
    )


conv_weight_groups = sorted(
    conv_weight_groups,
    key=conv_number
)


print(
    "\nConv weight groups:"
)

for x in conv_weight_groups:
    print(x)


if len(conv_layers) != len(conv_weight_groups):

    raise Exception(
        f"""
Mismatch!

Model convolution layers:
{len(conv_layers)}

Weight convolution layers:
{len(conv_weight_groups)}
"""
    )


# ==========================================================
# 4. Assign weights layer-by-layer
# ==========================================================

print(
    "\nLoading weights..."
)


with h5py.File(
    WEIGHTS_FILE,
    "r"
) as f:

    for layer, weight_name in zip(
        conv_layers,
        conv_weight_groups
    ):

        print(
            weight_name,
            "---->",
            layer.name
        )

        vars_group = (
            f["_layer_checkpoint_dependencies"]
            [weight_name]
            ["vars"]
        )

        kernel = vars_group["0"][:]

        bias = vars_group["1"][:]

        layer.set_weights(
            [
                kernel,
                bias
            ]
        )


print(
    "\nAll weights loaded successfully"
)


# ==========================================================
# 5. Save clean Keras 3 model
# ==========================================================

print(
    "\nSaving clean model..."
)


model.save(
    OUTPUT_FILE
)

print(
    "\n================================"
)

print(
    "SUCCESS"
)

print(
    "Created:",
    OUTPUT_FILE
)

print(
    "================================"
)
