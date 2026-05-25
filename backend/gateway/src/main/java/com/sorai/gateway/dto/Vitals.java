package com.sorai.gateway.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;

/** Patient vital signs as collected by the frontend form. */
public record Vitals(
        @Min(0)  @Max(120) int   age,
        @DecimalMin("30.0") @DecimalMax("45.0") double temp,   // °C
        @Min(20) @Max(300) int   hr,
        @Min(40) @Max(300) int   sbp,
        @Min(20) @Max(200) int   dbp,
        @Min(5)  @Max(80)  int   rr,
        @DecimalMin("50.0") @DecimalMax("100.0") double o2
) {}
