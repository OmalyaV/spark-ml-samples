package com.lohika.morning.ml.spark.driver.service.lyrics;

public enum Genre {

    POP("Pop", 0D),
    COUNTRY("Country", 1D),
    BLUES("Blues", 2D),
    JAZZ("Jazz", 3D),
    REGGAE("Reggae", 4D),
    ROCK("Rock", 5D),
    HIP_HOP("Hip Hop", 6D),
    LATIN("Latin", 7D),
    UNKNOWN("Unknown", -1D);

    private final String name;
    private final Double value;

    Genre(final String name, final Double value) {
        this.name = name;
        this.value = value;
    }

    public String getName() {
        return name;
    }

    public Double getValue() {
        return value;
    }

    /**
     * Maps a CSV genre string (e.g. "hip hop", "pop") to the corresponding enum.
     */
    public static Genre fromString(String genreString) {
        if (genreString == null) return UNKNOWN;
        switch (genreString.trim().toLowerCase()) {
            case "pop":     return POP;
            case "country": return COUNTRY;
            case "blues":   return BLUES;
            case "jazz":    return JAZZ;
            case "reggae":  return REGGAE;
            case "rock":    return ROCK;
            case "hip hop": return HIP_HOP;
            case "latin":   return LATIN;
            default:        return UNKNOWN;
        }
    }

    /**
     * Looks up a Genre by its numeric label value.
     */
    public static Genre fromValue(Double value) {
        for (Genre genre : Genre.values()) {
            if (genre.getValue().equals(value)) {
                return genre;
            }
        }
        return UNKNOWN;
    }
}
