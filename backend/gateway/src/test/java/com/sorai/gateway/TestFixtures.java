package com.sorai.gateway;

import com.sorai.gateway.dto.PredictRequest;
import com.sorai.gateway.dto.Vitals;

import java.util.List;

/** Shared sample data for gateway tests. */
public final class TestFixtures {

    private TestFixtures() {}

    public static PredictRequest criticalRequest() {
        return new PredictRequest(
                new Vitals(67, 38.2, 118, 95, 62, 22, 94.0),
                "Pacjent blady, spocony, ból w klatce piersiowej, duszność"
        );
    }

    public static PredictRequest stableRequest() {
        return new PredictRequest(
                new Vitals(30, 36.7, 72, 120, 80, 14, 99.0),
                "Wizyta kontrolna"
        );
    }

    public static String predictResponseJson() {
        return """
        {
          "finalCategory": 0,
          "confidence": 0.95,
          "modelPredictions": [
            {
              "modelName": "catboost",
              "category": 0,
              "probabilities": [0.95, 0.03, 0.01, 0.005, 0.005],
              "confidence": 0.95
            },
            {
              "modelName": "lightgbm",
              "category": 0,
              "probabilities": [0.92, 0.05, 0.02, 0.005, 0.005],
              "confidence": 0.92
            }
          ],
          "medgemma": {
            "category": 0,
            "confidence": 0.91,
            "reasoning": "Krytyczne parametry, ból w klatce — wymagana natychmiastowa interwencja.",
            "riskFlags": ["hipotensja", "tachykardia", "hipoksja"],
            "keyFindings": ["SBP=95", "HR=118", "SpO2=94%"]
          },
          "shapTop5": [
            {"feature": "triage_vital_sbp", "value": 0.342, "direction": "positive"},
            {"feature": "triage_vital_hr",  "value": 0.287, "direction": "positive"}
          ],
          "conflict": {
            "detected": false,
            "severity": "low",
            "alertDoctor": true,
            "message": "Krytyczne parametry"
          },
          "processingTimeMs": 142
        }
        """;
    }

    public static String healthResponseJson() {
        return """
        {
          "status": "ok",
          "modelsLoaded": ["catboost", "lightgbm"],
          "ollamaReady": true,
          "uptimeSeconds": 3600
        }
        """;
    }
}
