package com.sorai.gateway.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/** Mirrors backend.app.models.schemas.PredictResponse. */
public record PredictResponse(
        @JsonProperty("finalCategory")     int                    finalCategory,
        @JsonProperty("confidence")        double                 confidence,
        @JsonProperty("modelPredictions")  List<ModelPrediction>  modelPredictions,
        @JsonProperty("medgemma")          MedGemmaAssessment     medgemma,
        @JsonProperty("shapTop5")          List<ShapValue>        shapTop5,
        @JsonProperty("conflict")          ConflictInfo           conflict,
        @JsonProperty("processingTimeMs")  int                    processingTimeMs
) {
    public record ModelPrediction(
            @JsonProperty("modelName")     String         modelName,
            @JsonProperty("category")      int            category,
            @JsonProperty("probabilities") List<Double>   probabilities,
            @JsonProperty("confidence")    double         confidence
    ) {}

    public record ShapValue(
            @JsonProperty("feature")   String feature,
            @JsonProperty("value")     double value,
            @JsonProperty("direction") String direction
    ) {}

    public record MedGemmaAssessment(
            @JsonProperty("category")    int          category,
            @JsonProperty("confidence")  double       confidence,
            @JsonProperty("reasoning")   String       reasoning,
            @JsonProperty("riskFlags")   List<String> riskFlags,
            @JsonProperty("keyFindings") List<String> keyFindings
    ) {}

    public record ConflictInfo(
            @JsonProperty("detected")    boolean detected,
            @JsonProperty("severity")    String  severity,
            @JsonProperty("alertDoctor") boolean alertDoctor,
            @JsonProperty("message")     String  message
    ) {}
}
