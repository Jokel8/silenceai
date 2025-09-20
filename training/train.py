import os
import argparse
import numpy as np
from glob import glob
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

try:
    import tensorflow as tf
except Exception:
    tf = None


def build_simple_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def convert_to_tflite(input_model_path, output_tflite_path, fp16=False):
    """Convert a SavedModel directory or HDF5 model to TFLite. Returns True on success."""
    if tf is None:
        print("TensorFlow not available, cannot convert to TFLite")
        return False
    converter = None
    # If input is a directory assume SavedModel, else H5
    if os.path.isdir(input_model_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
    else:
        converter = tf.lite.TFLiteConverter.from_keras_model_file(input_model_path) if hasattr(tf.lite, 'TFLiteConverter') else None
        # fallback: load model then convert
        if converter is None:
            try:
                m = tf.keras.models.load_model(input_model_path)
                converter = tf.lite.TFLiteConverter.from_keras_model(m)
            except Exception as e:
                print('Failed to load model for conversion:', e)
                return False
    # set optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if fp16:
        converter.target_spec.supported_types = [tf.float16]
    try:
        tflite_model = converter.convert()
        with open(output_tflite_path, 'wb') as f:
            f.write(tflite_model)
        print('TFLite written to', output_tflite_path)
        return True
    except Exception as e:
        print('TFLite conversion failed:', e)
        return False


def main(args):
    # If convert-only mode: convert and exit
    if getattr(args, 'convert_only', False):
        if not args.input_model or not args.output_tflite:
            print('For convert-only mode, provide --input_model and --output_tflite')
            return
        convert_to_tflite(args.input_model, args.output_tflite, fp16=bool(args.tflite_fp16))
        return

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    out_dir = args.model_dir
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_dir = args.checkpoint_dir or os.path.join(out_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # infer classes
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes.sort()
    if not classes:
        print('No classes found in', train_dir)
        return

    le = LabelEncoder()
    le.fit(classes)
    joblib.dump(le, os.path.join(out_dir, 'label_encoder.joblib'))

    # data generators
    target_size = (args.width, args.height)
    batch = args.batch_size
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.05,
                                   height_shift_range=0.05, brightness_range=(0.8,1.2))
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(train_dir, target_size=target_size, batch_size=batch, class_mode='categorical')
    val_flow = val_gen.flow_from_directory(val_dir, target_size=target_size, batch_size=batch, class_mode='categorical')

    model = build_simple_cnn(input_shape=(args.height, args.width, 3), num_classes=len(classes))

    # Callbacks: ModelCheckpoint for best model and optional TensorBoard and EarlyStopping
    best_ckpt_path = os.path.join(checkpoint_dir, 'best.h5')
    cb = []
    monitor = args.monitor or 'val_loss'
    cb.append(callbacks.ModelCheckpoint(best_ckpt_path, save_best_only=True, monitor=monitor, mode='auto', verbose=1))
    if args.save_epochs:
        # save every epoch with metric in name
        epoch_pattern = os.path.join(checkpoint_dir, 'epoch_{epoch:02d}_' + monitor + '_{'+monitor+':.4f}.h5')
        try:
            cb.append(callbacks.ModelCheckpoint(epoch_pattern, save_best_only=False, verbose=0))
        except Exception:
            pass
    if args.tensorboard_logdir:
        tb_dir = args.tensorboard_logdir
        os.makedirs(tb_dir, exist_ok=True)
        cb.append(callbacks.TensorBoard(log_dir=tb_dir))
    cb.append(callbacks.EarlyStopping(patience=args.patience, restore_best_weights=True, monitor=monitor))

    model.fit(train_flow, epochs=args.epochs, validation_data=val_flow, callbacks=cb)

    # Save final and export best
    final_h5 = os.path.join(out_dir, 'final.h5')
    model.save(final_h5)
    print('Final model saved to', final_h5)

    # If training restored best weights, current model should be best; ensure best.h5 exists
    if os.path.exists(best_ckpt_path):
        exported_h5 = os.path.join(out_dir, 'best_model.h5')
        try:
            # copy best checkpoint to out_dir
            import shutil
            shutil.copy2(best_ckpt_path, exported_h5)
            print('Best checkpoint copied to', exported_h5)
        except Exception as e:
            print('Failed to copy best checkpoint:', e)

    # also save a SavedModel directory for deployment
    saved_dir = os.path.join(out_dir, 'best_saved_model')
    try:
        model.save(saved_dir, save_format='tf')
        print('SavedModel exported to', saved_dir)
    except Exception as e:
        print('Failed to export SavedModel:', e)

    # Optional TFLite conversion
    if args.tflite:
        tflite_out = os.path.join(out_dir, 'best_model.tflite')
        success = convert_to_tflite(saved_dir if os.path.isdir(saved_dir) else exported_h5, tflite_out, fp16=bool(args.tflite_fp16))
        if success:
            print('TFLite model saved to', tflite_out)

    print('Training complete. Models and checkpoints saved in', out_dir)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    # Primary (canonical) names
    p.add_argument('--data_dir', dest='data_dir', default='data', help='dataset root with train/val/test')
    p.add_argument('--model_dir', dest='model_dir', default='models', help='output directory')
    p.add_argument('--checkpoint_dir', dest='checkpoint_dir', default=None, help='checkpoint directory (overrides default)')

    # Backwards-compatible aliases (used by GUI and older callers)
    p.add_argument('--data', dest='data_dir', help=argparse.SUPPRESS)
    p.add_argument('--model', dest='model_dir', help=argparse.SUPPRESS)
    p.add_argument('--checkpoint', dest='checkpoint_dir', help=argparse.SUPPRESS)

    p.add_argument('--epochs', type=int, default=10)
    # accept both --batch_size and --batch-size
    p.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    p.add_argument('--batch-size', dest='batch_size', type=int, help=argparse.SUPPRESS)

    p.add_argument('--width', type=int, default=210)
    p.add_argument('--height', type=int, default=300)
    p.add_argument('--save_epochs', action='store_true', help='Also save model each epoch')
    p.add_argument('--monitor', default='val_loss', help='Metric to monitor for best checkpoint')
    p.add_argument('--tensorboard_logdir', default=None, help='If set, write TensorBoard logs here')
    p.add_argument('--patience', type=int, default=5, help='EarlyStopping patience')
    # tflite options
    p.add_argument('--tflite', action='store_true', help='Convert best model to TFLite after training')
    p.add_argument('--tflite_fp16', action='store_true', help='Use float16 quantization for TFLite')
    # convert only mode
    p.add_argument('--convert_only', action='store_true', help='Only convert an existing model to TFLite (use with --input_model --output_tflite)')
    p.add_argument('--input_model', default=None, help='Input model (SavedModel dir or .h5) for convert-only')
    p.add_argument('--output_tflite', default=None, help='Output .tflite path for convert-only')

    args = p.parse_args()
    main(args)
