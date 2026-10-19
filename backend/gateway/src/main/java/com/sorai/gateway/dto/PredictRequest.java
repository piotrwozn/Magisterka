package com.sorai.gateway.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

/**
 * Mirrors backend.app.models.schemas.PredictRequest (camelCase JSON wire format).
 * sex / arrivalMode / o2Device are optional (null = unknown, not sent to the ML service).
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record PredictRequest(
        @NotNull @Valid Vitals vitals,

        @Size(max = 2000)
        @JsonProperty("clinicalNote")
        String clinicalNote,

        @Pattern(regexp = "[MF]")
        @JsonProperty("sex")
        String sex,

        @Size(max = 40)
        @JsonProperty("arrivalMode")
        String arrivalMode,

        @JsonProperty("o2Device")
        Boolean o2Device
) {
    /** Request without the optional patient context. */
    public PredictRequest(Vitals vitals, String clinicalNote) {
        this(vitals, clinicalNote, null, null, null);
    }
}
