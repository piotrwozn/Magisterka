package com.sorai.gateway.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public record HealthStatus(
        @JsonProperty("status")         String       status,
        @JsonProperty("modelsLoaded")   List<String> modelsLoaded,
        @JsonProperty("ollamaReady")    boolean      ollamaReady,
        @JsonProperty("uptimeSeconds")  long         uptimeSeconds
) {}
