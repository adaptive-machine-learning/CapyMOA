# MOABridge
Python wrapper for MOA to allow efficient use of existing algorithms with a more modern API

To use the project
1. Set the config.ini to point to the correct path of the moa.jar
2. Make sure JAVA_HOME is set
3. The requirements are missing river and jpype
4. ```pip install jpype1``` (yes, there is a 1 there), and to run the river experiments you need to ```pip install river```

# TODO (updated 26/09/2023)
1. [Bug] Why the last window (with the remaining instances) is not displaying when using WindowClassificaitonEvaluator? 
2. [Tests/Examples] Add tests and examples using regression for prequential, test-then-train and windowed. Notebook: 
3. [New] Add prequential multiple streams and learners (will need to restart streams, provide timing information)
4. [Examples] Create a simple Demo notebook for Classification and Regression. 
5. [Tests/Examples] Double check if all notebooks still work
6. [New] Create the package structure.  
7. [New] Override the ```__str__``` or create a ```describe()``` method to return the learner+key hyperparameters (for plotting and identification purposes)
8. [New] Create ```__str__``` for the ```Stream()``` classes (to identify the dataset). 
