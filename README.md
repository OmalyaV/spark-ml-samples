# Sample Application for "Introduction to ML with Apache Spark MLlib" Presentation

## Presentation
Link to the presentation: http://www.slideshare.net/tmatyashovsky/introduction-to-ml-with-apache-spark-mllib

## Idea
Create few examples to demonstrate regression, classification and clustering to Java developers.
Main focus is on feature extraction and creation of interesting ML pipelines. 

### DOU Dataset
DOU (http://dou.ua) stands for developers.org.ua is a main hub for Ukrainian developers.
It provides anonymous survey for getting information about Ukrainian engineers, their salary, experience, English level, etc. 

#### DOU Dataset Regression
Given poll results predict salary based on experience, English level and programming language.

Nuances:
* English level is string, should be converted to numeric, e.g. 0…5
* Languages are strings, should be exploded to 18 booleans, e.g. java=0|1, python=0|1, etc.
* Sparse vector [21, [0, 1, 13], [3, 4, 1]] is more preferable

#### DOU Dataset Clustering
Given poll results predict level (junior, middle, senior) based on experience and English level.

Nuances:
* English level is string, should be converted to numeric, e.g. 0…5
* 1$ difference in salary is not as significant as 1 year of experience, so data should be scaled before clustering
* Dense vector is ok  

### Multi-Genre Classifier (Mendeley Dataset)
Given song lyrics, recognize the genre as one of 7 classes: Pop, Country, Blues, Jazz, Reggae, Rock, Hip Hop.

Strategy:
* Dataset: `tcc_ceds_music.csv` (Mendeley dataset with pre-processed lyrics)
* Train/Test Split: 80% / 20%
* Pipeline: `StringIndexer` -> `Tokenizer` -> `StopWordsRemover` -> `HashingTF` -> `IDF` -> `NaiveBayes`
* Evaluation: `MulticlassClassificationEvaluator` reporting Accuracy, F1, Precision, and Recall.

Includes a **Web-based UI** for real-time predictions with Canvas-based charts (Bar Graph, Donut Pie Chart).

### MNIST Dataset
Given set of images recognize digits.

Nuances:
* Tranform images into training examples

## Build and Run

### Prerequisites
* **Java 8** is required to build and run this project. 
* Ensure your `JAVA_HOME` points to a JDK 1.8 installation (e.g., Amazon Corretto 8).

### Build
To compile the project and build the JAR artifacts (excluding tests):
```sh
# On Windows
$env:JAVA_HOME="C:\Program Files\Amazon Corretto\jdk1.8.0_482"
.\gradlew build -x test

# On Linux/macOS
export JAVA_HOME=/path/to/java-8
./gradlew build -x test
```

### Run and Access the Web UI
The application includes an embedded Jetty web server and a static UI wrapper served by Spring Boot. To start it, run the packaged `api-1.0-SNAPSHOT.jar`:

```sh
java -jar api/build/libs/api-1.0-SNAPSHOT.jar
```

Once the consol logs indicate the application has started (e.g., `Started ApplicationConfiguration in...`), open your web browser and navigate to:

**http://localhost:9090**

You can then test the **Multi-Genre Classifier**:
1. Open the UI and paste song lyrics.
2. Click **Classify Genre** (the system will automatically train the model on the first request).
3. View the prediction and the corresponding visual analytics charts!
