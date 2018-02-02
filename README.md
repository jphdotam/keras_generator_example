# keras_generator_example

An example of a Keras generator which reads three-dimensional npy files (e.g. of videos) and labels using a directory format

This has been adapted from the great example at: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

Unlike shervinea's example, this method loads data using a simple directory structure rather than a dictionary of IDs and labels.

## Usage

1) Structure data in directories, like the following example:

```
data/
├── train/
    ├── label1/
        ├── example1.npy
        ├── example7.npy
        ├── example10.npy
    ├── label2/
        ├── example4.npy
        ├── example8.npy
        ├── example11.npy
├── test/
    ├── label1/
        ├── example2.npy
        ├── example5.npy
        ├── example9.npy
    ├── label2/
        ├── example6.npy
        ├── example11.npy
        ├── example12.npy
```

2) Initialise the generator:

If the videos are e.g. 128 x 128 pixels and 20 frames long:

```python
batch_size = 16
generator = VideoGenerator(width=128, height=128, frames=20,
                           batch_size=batch_size, shuffle=True,
                           inputdir="./data", fileext=".npy")
```

3) Create generators for training and evaluation:

```python
training_generator = generator.generate(train_or_test_or_eval="train")
training_steps_per_epoch = len(generator.filenames_train) // batch_size
testing_generator = generator.generate(train_or_test_or_eval="test")
testing_steps_per_epoch = len(generator.filenames_test) // batch_size
```

4) Fit a Keras model to a generator:

```python
model.fit_generator(generator=training_generator,
                     steps_per_epoch=training_steps_per_epoch,
                     verbose=2,
                     max_queue_size=10,
                     validation_data=testing_generator,
                     validation_steps=testing_steps_per_epoch,
                     epochs=epochs)
```
