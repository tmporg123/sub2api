package service

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestForceOpenAIMaxEffortAndPriorityTier(t *testing.T) {
	gin.SetMode(gin.TestMode)

	openaiAccount := &Account{Platform: PlatformOpenAI, Type: AccountTypeAPIKey}
	grokAccount := &Account{Platform: PlatformGrok, Type: AccountTypeOAuth}

	t.Run("responses injects max and priority", func(t *testing.T) {
		c := newGinContextWithPath("/v1/responses")
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, []byte(`{"model":"gpt-5.4","input":"hi","reasoning":{"effort":"low"},"service_tier":"flex"}`))
		require.Equal(t, "max", gjson.GetBytes(got, "reasoning.effort").String())
		require.Equal(t, "priority", gjson.GetBytes(got, "service_tier").String())
		require.Equal(t, "gpt-5.4", gjson.GetBytes(got, "model").String())
	})

	t.Run("missing fields are injected", func(t *testing.T) {
		c := newGinContextWithPath("/openai/v1/responses")
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, []byte(`{"model":"gpt-5.4","input":"hi"}`))
		require.Equal(t, "max", gjson.GetBytes(got, "reasoning.effort").String())
		require.Equal(t, "priority", gjson.GetBytes(got, "service_tier").String())
	})

	t.Run("chat completions uses reasoning_effort", func(t *testing.T) {
		c := newGinContextWithPath("/v1/chat/completions")
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, []byte(`{"model":"gpt-5.4","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"medium","service_tier":"flex"}`))
		require.Equal(t, "max", gjson.GetBytes(got, "reasoning_effort").String())
		require.Equal(t, "priority", gjson.GetBytes(got, "service_tier").String())
		require.False(t, gjson.GetBytes(got, "reasoning.effort").Exists())
	})

	t.Run("compact path is left unchanged", func(t *testing.T) {
		c := newGinContextWithPath("/v1/responses/compact")
		body := []byte(`{"model":"gpt-5.4","input":"hi","reasoning":{"effort":"low"},"service_tier":"flex"}`)
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, body)
		require.Equal(t, string(body), string(got))
	})

	t.Run("compact substring in prefixed path is skipped", func(t *testing.T) {
		c := newGinContextWithPath("/openai/v1/responses/compact")
		body := []byte(`{"model":"gpt-5.4","reasoning":{"effort":"high"}}`)
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, body)
		require.Equal(t, string(body), string(got))
	})

	t.Run("grok accounts are skipped", func(t *testing.T) {
		c := newGinContextWithPath("/v1/responses")
		body := []byte(`{"model":"grok-4","input":"hi","reasoning":{"effort":"low"},"service_tier":"flex"}`)
		got := forceOpenAIMaxEffortAndPriorityTier(c, grokAccount, body)
		require.Equal(t, string(body), string(got))
	})

	t.Run("ws response.create is rewritten", func(t *testing.T) {
		c := newGinContextWithPath("/v1/responses")
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, []byte(`{"type":"response.create","model":"gpt-5.4","service_tier":"flex","reasoning":{"effort":"high"}}`))
		require.Equal(t, "max", gjson.GetBytes(got, "reasoning.effort").String())
		require.Equal(t, "priority", gjson.GetBytes(got, "service_tier").String())
		require.Equal(t, "response.create", gjson.GetBytes(got, "type").String())
	})

	t.Run("ws non create frames are left unchanged", func(t *testing.T) {
		c := newGinContextWithPath("/v1/responses")
		body := []byte(`{"type":"response.cancel","service_tier":"flex"}`)
		got := forceOpenAIMaxEffortAndPriorityTier(c, openaiAccount, body)
		require.Equal(t, string(body), string(got))
	})
}

func TestForceOpenAIMaxEffortAndPriorityTierOnMap(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c := newGinContextWithPath("/v1/responses")
	account := &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth}

	reqBody := map[string]any{
		"model":        "gpt-5.4",
		"input":        "hi",
		"service_tier": "flex",
		"reasoning":    map[string]any{"effort": "low", "summary": "auto"},
	}
	forceOpenAIMaxEffortAndPriorityTierOnMap(c, account, reqBody)
	require.Equal(t, "priority", reqBody["service_tier"])
	reasoning := reqBody["reasoning"].(map[string]any)
	require.Equal(t, "max", reasoning["effort"])
	require.Equal(t, "auto", reasoning["summary"])
}

func newGinContextWithPath(path string) *gin.Context {
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, path, nil)
	return c
}
