package com.lohika.morning.ml.spark.driver.service.lyrics;

import java.util.Map;

public class GenrePrediction {

    private String genre;
    private Map<String, Double> probabilities;

    public GenrePrediction(String genre, Map<String, Double> probabilities) {
        this.genre = genre;
        this.probabilities = probabilities;
    }

    public GenrePrediction(String genre) {
        this.genre = genre;
    }

    public String getGenre() {
        return genre;
    }

    public Map<String, Double> getProbabilities() {
        return probabilities;
    }
}
