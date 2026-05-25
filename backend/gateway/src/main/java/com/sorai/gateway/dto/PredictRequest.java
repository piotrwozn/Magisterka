package com.sorai.gateway.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

/** Mirrors backend.app.models.schemas.PredictRequest (camelCase JSON wire format). */
public record PredictRequest(
        @NotNull @Valid Vitals vitals,

        @Size(max = 2000)
        @JsonProperty("clinicalNote")
        String clinicalNote
) {}
