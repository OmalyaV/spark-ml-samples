package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.apache.spark.sql.SparkSession;

@Component("MendeleyLogisticRegressionPipeline")
public class MendeleyLogisticRegressionPipeline implements LyricsPipeline {

    @Autowired
    private SparkSession sparkSession;

    @Value("${mendeley.csv.file.path}")
    private String csvFilePath;

    @Value("${lyrics.model.directory.path:models}")
    private String modelDirectoryPath;

    // Cached model
    private volatile CrossValidatorModel cachedModel;
    private volatile PipelineModel cachedBestModel;

    @Override
    public CrossValidatorModel classify() {
        Dataset<Row> data = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(csvFilePath);

        data = data.select("genre", "lyrics").na().drop(new String[]{"lyrics", "genre"});

        System.out.println("\n===== Dataset Overview =====");
        data.groupBy("genre").count().show();

        StringIndexer genreIndexer = new StringIndexer()
                .setInputCol("genre")
                .setOutputCol("label")
                .setHandleInvalid("keep");

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("lyrics")
                .setOutputCol("words");

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered_words");

        // 1. NGram: Create bi-grams for context (e.g. "hip hop", "love you")
        NGram bigram = new NGram().setN(2).setInputCol("filtered_words").setOutputCol("bigrams");

        // Merge unigrams and bigrams
        // But to keep it simple and avoid custom transformers, we'll just use CountVectorizer on unigrams first, 
        // as CountVectorizer alone significantly outperforms HashingTF for text.
        
        // 2. CountVectorizer explicitly learns the vocabulary instead of hashing, avoiding collisions.
        CountVectorizer countVectorizer = new CountVectorizer()
                .setInputCol("filtered_words")
                .setOutputCol("raw_features")
                .setVocabSize(15000)
                .setMinDF(3.0); // words must appear in at least 3 documents

        IDF idf = new IDF()
                .setInputCol("raw_features")
                .setOutputCol("features");

        // 3. Logistic Regression is often superior to NaiveBayes for multi-class text
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(30)
                .setElasticNetParam(0.0); // Ridge regression

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                genreIndexer,
                tokenizer,
                stopWordsRemover,
                countVectorizer,
                idf,
                lr
        });

        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainingData = splits[0].cache();
        Dataset<Row> testData = splits[1].cache();

        // Tune the regularization parameter for Logistic Regression
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{0.01, 0.1, 1.0})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator()
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setMetricName("accuracy"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3); // Reduced folds to 3 to speed up training

        CrossValidatorModel model = crossValidator.fit(trainingData);
        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Dataset<Row> predictions = bestModel.transform(testData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        double accuracy    = evaluator.setMetricName("accuracy").evaluate(predictions);
        double f1          = evaluator.setMetricName("f1").evaluate(predictions);

        System.out.println("\n===== Test Set Evaluation (Logistic Regression) =====");
        System.out.println("Accuracy:           " + String.format("%.4f", accuracy));
        System.out.println("F1 Score:           " + String.format("%.4f", f1));

        this.cachedModel = model;
        this.cachedBestModel = bestModel;

        try {
            model.write().overwrite().save(modelDirectoryPath + "/mendeley-lr-model");
            System.out.println("Model saved successfully to: " + modelDirectoryPath + "/mendeley-lr-model");
        } catch (Exception e) {
            System.err.println("Failed to save model: " + e.getMessage());
        }

        return model;
    }

    private void loadOrTrainModel() {
        if (cachedBestModel != null) {
            return;
        }
        
        String path = modelDirectoryPath + "/mendeley-lr-model";
        if (Files.exists(Paths.get(path))) {
            try {
                System.out.println("Loading existing model from disk at: " + path);
                this.cachedModel = CrossValidatorModel.load(path);
                this.cachedBestModel = (PipelineModel) this.cachedModel.bestModel();
                System.out.println("Model loaded successfully.");
                return;
            } catch (Exception e) {
                System.err.println("Failed to load model from disk: " + e.getMessage() + ". Retraining...");
            }
        }
        
        System.out.println("No existing model found or load failed. Training a new model...");
        classify();
    }

    @Override
    public GenrePrediction predict(String unknownLyrics) {
        loadOrTrainModel();

        Dataset<Row> lyricsDF = sparkSession.createDataFrame(
                Collections.singletonList(RowFactory.create("unknown", unknownLyrics)),
                new StructType(new StructField[]{
                        DataTypes.createStructField("genre", DataTypes.StringType, true),
                        DataTypes.createStructField("lyrics", DataTypes.StringType, true)
                })
        );

        Dataset<Row> result = cachedBestModel.transform(lyricsDF);
        Row predictionRow = result.first();
        Double predictionIndex = predictionRow.getAs("prediction");

        StringIndexerModel indexerModel = (StringIndexerModel) cachedBestModel.stages()[0];
        String[] labels = indexerModel.labels();
        String predictedGenre = (predictionIndex.intValue() < labels.length) ? labels[predictionIndex.intValue()] : "Unknown";

        Map<String, Double> probabilities = new HashMap<>();
        if (Arrays.asList(result.columns()).contains("probability")) {
            DenseVector prob = predictionRow.getAs("probability");
            for (int i = 0; i < labels.length && i < prob.size(); i++) {
                probabilities.put(labels[i], prob.apply(i));
            }
        }

        Genre genre = Genre.fromString(predictedGenre);
        return new GenrePrediction(genre.getName(), probabilities);
    }

    @Override
    public Map<String, Object> getModelStatistics(CrossValidatorModel model) {
        Map<String, Object> stats = new HashMap<>();
        Arrays.sort(model.avgMetrics());
        stats.put("Best model accuracy", model.avgMetrics()[model.avgMetrics().length - 1]);
        PipelineModel bestModel = (PipelineModel) model.bestModel();
        CountVectorizerModel cv = (CountVectorizerModel) bestModel.stages()[3];
        stats.put("Vocab size", cv.vocabulary().length);
        return stats;
    }
}
