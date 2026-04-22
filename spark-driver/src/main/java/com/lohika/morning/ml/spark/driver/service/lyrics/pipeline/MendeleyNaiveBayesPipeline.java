package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.driver.service.lyrics.Genre;
import com.lohika.morning.ml.spark.driver.service.lyrics.GenrePrediction;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
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

@Component("MendeleyNaiveBayesPipeline")
public class MendeleyNaiveBayesPipeline implements LyricsPipeline {

    @Autowired
    private SparkSession sparkSession;

    @Value("${mendeley.csv.file.path}")
    private String csvFilePath;

    @Value("${lyrics.model.directory.path:models}")
    private String modelDirectoryPath;

    // ── Cached model so predict() doesn't retrain every time ──────────
    private volatile CrossValidatorModel cachedModel;
    private volatile PipelineModel cachedBestModel;

    @Override
    public CrossValidatorModel classify() {
        // ── 1. Load data ──────────────────────────────────────────────────
        Dataset<Row> data = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(csvFilePath);

        // Keep only the columns we need.
        data = data.select("genre", "lyrics");

        // Drop rows with null lyrics.
        data = data.na().drop(new String[]{"lyrics", "genre"});

        System.out.println("\n===== Dataset Overview =====");
        System.out.println("Total rows: " + data.count());
        data.groupBy("genre").count().show();

        // ── 2. Label encoding ─────────────────────────────────────────────
        StringIndexer genreIndexer = new StringIndexer()
                .setInputCol("genre")
                .setOutputCol("label")
                .setHandleInvalid("keep");

        // ── 3. Tokenize (lyrics are already space-separated tokens) ───────
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("lyrics")
                .setOutputCol("words");

        // ── 4. Remove stop words ──────────────────────────────────────────
        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered_words");

        // ── 5. TF-IDF feature extraction ──────────────────────────────────
        HashingTF hashingTF = new HashingTF()
                .setInputCol("filtered_words")
                .setOutputCol("raw_features")
                .setNumFeatures(8192);

        IDF idf = new IDF()
                .setInputCol("raw_features")
                .setOutputCol("features");

        // ── 6. Classifier ─────────────────────────────────────────────────
        NaiveBayes naiveBayes = new NaiveBayes();

        // ── 7. Build pipeline ─────────────────────────────────────────────
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                genreIndexer,
                tokenizer,
                stopWordsRemover,
                hashingTF,
                idf,
                naiveBayes
        });

        // ── 8. 80/20 Train-Test split ─────────────────────────────────────
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        System.out.println("Training set size: " + trainingData.count());
        System.out.println("Test set size:     " + testData.count());

        // ── 9. Cross-validation on training set ───────────────────────────
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(hashingTF.numFeatures(), new int[]{4096, 8192})
                .build();

        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator()
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setMetricName("accuracy"))
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5);

        CrossValidatorModel model = crossValidator.fit(trainingData);

        // ── 10. Evaluate on the held-out test set ─────────────────────────
        PipelineModel bestModel = (PipelineModel) model.bestModel();
        Dataset<Row> predictions = bestModel.transform(testData);

        // Print some example predictions.
        System.out.println("\n===== Sample Predictions =====");
        predictions.select("genre", "lyrics", "label", "prediction").show(20, true);

        // Compute metrics.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");

        double accuracy    = evaluator.setMetricName("accuracy").evaluate(predictions);
        double f1          = evaluator.setMetricName("f1").evaluate(predictions);
        double precision   = evaluator.setMetricName("weightedPrecision").evaluate(predictions);
        double recall      = evaluator.setMetricName("weightedRecall").evaluate(predictions);

        System.out.println("\n===== Test Set Evaluation =====");
        System.out.println("Accuracy:           " + String.format("%.4f", accuracy));
        System.out.println("F1 Score:           " + String.format("%.4f", f1));
        System.out.println("Weighted Precision: " + String.format("%.4f", precision));
        System.out.println("Weighted Recall:    " + String.format("%.4f", recall));

        // Show per-genre prediction counts.
        System.out.println("\n===== Prediction Distribution =====");
        predictions.groupBy("genre", "prediction").count().orderBy("genre", "prediction").show(50, false);

        // Cache the model for predict().
        this.cachedModel = model;
        this.cachedBestModel = bestModel;

        return model;
    }

    @Override
    public GenrePrediction predict(String unknownLyrics) {
        // Ensure we have a trained model.
        if (cachedBestModel == null) {
            classify();
        }

        // Create a single-row DataFrame with the lyrics.
        Dataset<Row> lyricsDF = sparkSession.createDataFrame(
                Collections.singletonList(
                        RowFactory.create("unknown", unknownLyrics)
                ),
                new StructType(new StructField[]{
                        DataTypes.createStructField("genre", DataTypes.StringType, true),
                        DataTypes.createStructField("lyrics", DataTypes.StringType, true)
                })
        );

        Dataset<Row> result = cachedBestModel.transform(lyricsDF);
        Row predictionRow = result.first();
        Double predictionIndex = predictionRow.getAs("prediction");

        // Get the StringIndexer labels to map index back to genre name.
        StringIndexerModel indexerModel = (StringIndexerModel) cachedBestModel.stages()[0];
        String[] labels = indexerModel.labels();
        String predictedGenre = (predictionIndex.intValue() < labels.length)
                ? labels[predictionIndex.intValue()]
                : "Unknown";

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

        java.util.Arrays.sort(model.avgMetrics());
        stats.put("Best model accuracy", model.avgMetrics()[model.avgMetrics().length - 1]);

        PipelineModel bestModel = (PipelineModel) model.bestModel();
        HashingTF tf = (HashingTF) bestModel.stages()[3];
        stats.put("Num features (HashingTF)", tf.getNumFeatures());

        System.out.println("\n===== Model Statistics =====");
        stats.forEach((k, v) -> System.out.println(k + " = " + v));

        return stats;
    }
}
