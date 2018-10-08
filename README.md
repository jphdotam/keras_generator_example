# keras_generator_example

An example of a Keras generator which reads three-dimensional npy files (e.g. of videos) and labels using a directory format

It seems to be quite efficient (I get ~ 98% GPU usage on a 1080 Ti, so there's no CPU bottlenecking; I will likely introduce data augmentation to this with time).

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

If the videos are e.g. 299 x 299 pixels in RGB and 10 frames long in npy or npz format:

```python
self.generator = VideoGenerator(train_dir=train_dir,
                                test_dir=test_dir,
                                dims=(10, 299, 299, 3),
                                batch_size=16,
                                shuffle=True,
                                file_ext=".np*")
```

3) Create generators for training and evaluation:

```python
self.training_generator = self.generator.generate(train_or_test='train')
self.training_steps_per_epoch = len(self.generator.filenames_train) // self.batch_size
self.testing_generator = self.generator.generate(train_or_test="test")
self.testing_steps_per_epoch = len(self.generator.filenames_test) // self.batch_size
```

4) Fit a Keras model to a generator:

```python
self.model.fit_generator(self.training_generator,
                         steps_per_epoch=self.training_steps_per_epoch,
                         epochs=epochs,
                         validation_data=self.testing_generator,
                         validation_steps=self.testing_steps_per_epoch)
```
