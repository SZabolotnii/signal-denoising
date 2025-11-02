# signal-denoising

## TODO list:
1. ~~add generation of polygausse noise~~
2. refactor code to generate signals with different lenghts
   - current assumption is that all signals have the same length. it is needed for autoencoder architecture, or all 
   signals should be padded to the same length.
3. ~~write separate classes for training and models itself~~
4. generate big dataset (about 10k examples to start)
5. train models on the dataset
6. create comparison table with metrics
7. Use parameters `nperseg = 128, 
noverlap = 96` for all models and compare results

## Articles
Idea to write 2 articles:
- compare different approaches (analytical, transformers and autoencoders)
- compare 3 different autoencoders