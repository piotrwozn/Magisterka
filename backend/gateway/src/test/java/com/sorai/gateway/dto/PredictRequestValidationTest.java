package com.sorai.gateway.dto;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/** Validates that Bean Validation rules match the frontend's Vitals input range. */
class PredictRequestValidationTest {

    private static Validator validator;
    private static ObjectMapper mapper;

    @BeforeAll
    static void setUp() {
        validator = Validation.buildDefaultValidatorFactory().getValidator();
        mapper    = new ObjectMapper();
    }

    @Test
    @DisplayName("Valid request passes all constraints")
    void validRequestPasses() {
        var req = new PredictRequest(
                new Vitals(40, 36.7, 80, 120, 80, 16, 99.0),
                "Test"
        );
        Set<ConstraintViolation<PredictRequest>> v = validator.validate(req);
        assertThat(v).isEmpty();
    }

    @ParameterizedTest(name = "age={0} should fail")
    @CsvSource({"-1", "150", "999"})
    @DisplayName("Age out of [0..120] is rejected")
    void invalidAgeFails(int age) {
        var req = new PredictRequest(
                new Vitals(age, 36.7, 80, 120, 80, 16, 99.0),
                "Test"
        );
        assertThat(validator.validate(req)).isNotEmpty();
    }

    @ParameterizedTest(name = "temp={0}°C should fail")
    @CsvSource({"20.0", "50.0", "29.9", "45.1"})
    void invalidTempFails(double temp) {
        var req = new PredictRequest(
                new Vitals(40, temp, 80, 120, 80, 16, 99.0),
                "Test"
        );
        assertThat(validator.validate(req)).isNotEmpty();
    }

    @ParameterizedTest(name = "o2={0}% should fail")
    @CsvSource({"40.0", "110.0", "49.9", "100.1"})
    void invalidO2Fails(double o2) {
        var req = new PredictRequest(
                new Vitals(40, 36.7, 80, 120, 80, 16, o2),
                "Test"
        );
        assertThat(validator.validate(req)).isNotEmpty();
    }

    @Test
    @DisplayName("Clinical note >2000 chars is rejected")
    void noteTooLongFails() {
        String longNote = "x".repeat(2001);
        var req = new PredictRequest(
                new Vitals(40, 36.7, 80, 120, 80, 16, 99.0),
                longNote
        );
        assertThat(validator.validate(req)).isNotEmpty();
    }

    @Test
    @DisplayName("JSON deserialization accepts camelCase from frontend")
    void deserializesCamelCase() throws Exception {
        String json = """
                {
                  "vitals": {"age":40,"temp":36.7,"hr":80,"sbp":120,"dbp":80,"rr":16,"o2":99.0},
                  "clinicalNote": "ból w klatce"
                }
                """;
        PredictRequest parsed = mapper.readValue(json, PredictRequest.class);
        assertThat(parsed.vitals().age()).isEqualTo(40);
        assertThat(parsed.clinicalNote()).isEqualTo("ból w klatce");
    }
}
