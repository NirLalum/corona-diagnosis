# Machine learning based predictions of covid 19 diagnosis based on symptoms

## Built a full machine learning pipeline to predict covid 19 diagnosis: data pre-processing, data modeling and evaluating and post processing:
* data exploration
* data imputation
* upsampling to overcome the imbalanced data issue (negative - 93.3%, positive - 5.3%, other - 1.4%)
* removed non informative features
* new features were created
* data modeling using different algorithms like: random forest, xgboost and a deep neural network.
* evaluating the data using different metrics
* data post-processing for understanding where the model was wrong (explored the missclassified examples)

## The pipeline was implemented in python, using libraries such as pandas, sci-kit learn, Pytorch and Matplotlib.

## Best results:
 ![results_corona](https://user-images.githubusercontent.com/58294441/143613829-29a4e491-cd9a-4f46-b3d1-ceddf897e2a5.JPG)
 
 ## Some notes about the result:
 * got these results using a simple deep neural network with adam optimizer and a batch size of 2^14.
 * a dummy classifier gave accuracy of about 94% where all the rest metrics were 0 (due to the imbalanced data issue).
 * The main goal of the project is to locate the positive peoples because we don't want to spread the virus, so high recall is very important.
 * I got better recall with other models and other training data forms, but the precision was very low due to recall-precision trade off, so for now i think it's the best result, because low precision means that many people who were diagnosed as positive while they are not. We would like to avoid this.
 
 ## In the future:
 * I'd like to improve the precision while not harming the recall (and maybe even improve it).
 * Ideas for improving:
    * clustering the missclassified examples and understand better their nature.
    * using trained models like google's TabNet (there is allreay an implementaion of it but a GPU is needed)
    * add more data to the data set.
