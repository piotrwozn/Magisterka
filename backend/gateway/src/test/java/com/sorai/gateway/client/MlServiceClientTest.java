package com.sorai.gateway.client;

import com.sorai.gateway.TestFixtures;
import com.sorai.gateway.dto.HealthStatus;
import com.sorai.gateway.dto.PredictResponse;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/** Tests the WebClient-based MlServiceClient against a MockWebServer. */
class MlServiceClientTest {

    private MockWebServer server;
    private MlServiceClient client;

    @BeforeEach
    void setUp() throws IOException {
        server = new MockWebServer();
        server.start();

        WebClient webClient = WebClient.builder()
                .baseUrl(server.url("/").toString())
                .build();
        client = new MlServiceClient(webClient);
    }

    @AfterEach
    void tearDown() throws IOException {
        server.shutdown();
    }

    @Test
    @DisplayName("predict() forwards body and parses camelCase response")
    void predictForwardsAndParses() throws InterruptedException {
        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .setBody(TestFixtures.predictResponseJson()));

        StepVerifier.create(client.predict(TestFixtures.criticalRequest()))
                .assertNext(resp -> {
                    assertThat(resp.finalCategory()).isEqualTo(0);
                    assertThat(resp.confidence()).isCloseTo(0.95, org.assertj.core.data.Offset.offset(0.01));
                    assertThat(resp.modelPredictions()).hasSize(2);
                    assertThat(resp.modelPredictions().get(0).modelName()).isEqualTo("catboost");
                    assertThat(resp.conflict().alertDoctor()).isTrue();
                    assertThat(resp.medgemma().riskFlags()).contains("hipotensja");
                    assertThat(resp.shapTop5()).isNotEmpty();
                    assertThat(resp.processingTimeMs()).isEqualTo(142);
                })
                .verifyComplete();

        RecordedRequest req = server.takeRequest(2, TimeUnit.SECONDS);
        assertThat(req).isNotNull();
        assertThat(req.getPath()).isEqualTo("/api/v1/predict");
        assertThat(req.getMethod()).isEqualTo("POST");
        assertThat(req.getBody().readUtf8()).contains("\"clinicalNote\"");
    }

    @Test
    @DisplayName("predict() retries on transient 500, succeeds on second attempt")
    void predictRetriesOnError() {
        server.enqueue(new MockResponse().setResponseCode(500));
        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .setBody(TestFixtures.predictResponseJson()));

        StepVerifier.create(client.predict(TestFixtures.criticalRequest()))
                .assertNext(resp -> assertThat(resp.finalCategory()).isEqualTo(0))
                .verifyComplete();

        assertThat(server.getRequestCount()).isGreaterThanOrEqualTo(2);
    }

    @Test
    @DisplayName("predict() falls back when ML is permanently down")
    void predictFallbackOnPermanentError() {
        // Three failures will exhaust retries (default 3) and trigger fallback
        for (int i = 0; i < 5; i++) {
            server.enqueue(new MockResponse().setResponseCode(503));
        }

        StepVerifier.create(client.predict(TestFixtures.criticalRequest()))
                .expectError(MlServiceClient.MlServiceUnavailableException.class)
                .verify();
    }

    @Test
    @DisplayName("health() returns parsed HealthStatus")
    void healthEndpoint() {
        server.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .setBody(TestFixtures.healthResponseJson()));

        StepVerifier.create(client.health())
                .assertNext(h -> {
                    assertThat(h.status()).isEqualTo("ok");
                    assertThat(h.modelsLoaded()).containsExactly("catboost", "lightgbm");
                    assertThat(h.ollamaReady()).isTrue();
                    assertThat(h.uptimeSeconds()).isEqualTo(3600);
                })
                .verifyComplete();
    }
}
