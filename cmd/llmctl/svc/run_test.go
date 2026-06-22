/*
Copyright 2026 The RBG Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package svc

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/cli-runtime/pkg/genericclioptions"

	llmmeta "sigs.k8s.io/rbgs/cli/cmd/llmctl/svc/metadata"
)

// TestMain sets up an isolated test environment for all tests in this package.
// We set RBG_MODEL_CONFIG to a non-existent path to prevent loading user models,
// ensuring tests only use the built-in models.yaml definitions.
func TestMain(m *testing.M) {
	// Set RBG_MODEL_CONFIG to a non-existent path to skip user model loading
	// This ensures tests use only built-in models, making them deterministic
	_ = os.Setenv("RBG_MODEL_CONFIG", "/nonexistent/path/for/testing")
	os.Exit(m.Run())
}

// --- newRunCmd: command metadata ---

func TestNewRunCmd_UseAndShort(t *testing.T) {
	cf := genericclioptions.NewConfigFlags(true)
	cmd := newRunCmd(cf)
	assert.Equal(t, "run <name> <model-id> [flags]", cmd.Use)
	assert.NotEmpty(t, cmd.Short)
}

func TestNewRunCmd_ExactlyTwoArgs(t *testing.T) {
	cf := genericclioptions.NewConfigFlags(true)
	cmd := newRunCmd(cf)
	// no args should produce an error
	err := cmd.Args(cmd, []string{})
	require.Error(t, err)

	// one arg should also error
	err = cmd.Args(cmd, []string{"my-qwen"})
	require.Error(t, err)

	// three args should also error
	err = cmd.Args(cmd, []string{"my-qwen", "Qwen/Qwen3.5-0.8B", "extra"})
	require.Error(t, err)

	// exactly two args is fine
	err = cmd.Args(cmd, []string{"my-qwen", "Qwen/Qwen3.5-0.8B"})
	require.NoError(t, err)
}

// --- newRunCmd: flags exist with expected defaults ---

func TestNewRunCmd_FlagDefaults(t *testing.T) {
	cf := genericclioptions.NewConfigFlags(true)
	cmd := newRunCmd(cf)

	// --name flag should not exist (now positional arg)
	nameFlag := cmd.Flags().Lookup("name")
	assert.Nil(t, nameFlag)

	// --mode default is empty (first mode in model config is used)
	modeFlag := cmd.Flags().Lookup("mode")
	require.NotNil(t, modeFlag)
	assert.Equal(t, "", modeFlag.DefValue)

	// --replicas default is 1
	replicasFlag := cmd.Flags().Lookup("replicas")
	require.NotNil(t, replicasFlag)
	assert.Equal(t, "1", replicasFlag.DefValue)

	// --revision default is "main"
	revFlag := cmd.Flags().Lookup("revision")
	require.NotNil(t, revFlag)
	assert.Equal(t, "main", revFlag.DefValue)

	// --storage default is empty
	storageFlag := cmd.Flags().Lookup("storage")
	require.NotNil(t, storageFlag)
	assert.Equal(t, "", storageFlag.DefValue)

	// --engine default is empty
	engineFlag := cmd.Flags().Lookup("engine")
	require.NotNil(t, engineFlag)
	assert.Equal(t, "", engineFlag.DefValue)

	// --dry-run default is false
	dryRunFlag := cmd.Flags().Lookup("dry-run")
	require.NotNil(t, dryRunFlag)
	assert.Equal(t, "false", dryRunFlag.DefValue)

	// --env and --arg are StringArray, default empty
	envFlag := cmd.Flags().Lookup("env")
	require.NotNil(t, envFlag)

	argFlag := cmd.Flags().Lookup("arg")
	require.NotNil(t, argFlag)
}

// --- env-var parsing logic (mirrors run.go's inline SplitN logic) ---
// run.go: parts := strings.SplitN(env, "=", 2)
// We test the same logic directly.

func splitEnvVarTestHelper(env string) []string {
	return strings.SplitN(env, "=", 2)
}

func TestRunEnvVarParsing_ValidKeyValue(t *testing.T) {
	parts := splitEnvVarTestHelper("FOO=bar")
	require.Len(t, parts, 2)
	assert.Equal(t, "FOO", parts[0])
	assert.Equal(t, "bar", parts[1])
}

func TestRunEnvVarParsing_ValueWithEquals(t *testing.T) {
	// Value itself contains "=" — only the first "=" is the separator
	parts := splitEnvVarTestHelper("KEY=val=ue")
	require.Len(t, parts, 2)
	assert.Equal(t, "KEY", parts[0])
	assert.Equal(t, "val=ue", parts[1])
}

func TestRunEnvVarParsing_NoEqualsSign(t *testing.T) {
	// SplitN with n=2 returns single element when no separator
	parts := splitEnvVarTestHelper("NOEQUALS")
	assert.Len(t, parts, 1)
}

func TestRunEnvVarParsing_EmptyValue(t *testing.T) {
	parts := splitEnvVarTestHelper("EMPTY=")
	require.Len(t, parts, 2)
	assert.Equal(t, "EMPTY", parts[0])
	assert.Equal(t, "", parts[1])
}

// --- generateRBG ---

func TestGenerateRBG_DefaultMode_VLLMEngine(t *testing.T) {
	// Qwen/Qwen3.5-0.8B standard mode uses vllm with port 8000
	rbg, metadata, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "my-svc", rbg.Name)
	assert.Equal(t, "default", rbg.Namespace)

	// Check returned metadata
	assert.Equal(t, "vllm", metadata.Engine)
	assert.Equal(t, "standard", metadata.Mode)
	assert.Equal(t, int32(8000), metadata.Port)

	// Check metadata annotation
	var annotationMetadata llmmeta.RunMetadata
	err = json.Unmarshal([]byte(rbg.Annotations[llmmeta.RunCommandMetadataAnnotationKey]), &annotationMetadata)
	require.NoError(t, err)
	assert.Equal(t, "vllm", annotationMetadata.Engine)
	assert.Equal(t, "standard", annotationMetadata.Mode)
	assert.Equal(t, int32(8000), annotationMetadata.Port)
}

func TestGenerateRBG_LatencyMode_SGLangEngine(t *testing.T) {
	// Qwen/Qwen3.5-0.8B latency mode uses sglang with port 30000
	rbg, metadata, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Mode:     "latency",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	// Check returned metadata
	assert.Equal(t, "sglang", metadata.Engine)
	assert.Equal(t, "latency", metadata.Mode)
	assert.Equal(t, int32(30000), metadata.Port)

	// Check metadata annotation
	var annotationMetadata llmmeta.RunMetadata
	err = json.Unmarshal([]byte(rbg.Annotations[llmmeta.RunCommandMetadataAnnotationKey]), &annotationMetadata)
	require.NoError(t, err)
	assert.Equal(t, "sglang", annotationMetadata.Engine)
	assert.Equal(t, "latency", annotationMetadata.Mode)
	assert.Equal(t, int32(30000), annotationMetadata.Port)
}

func TestGenerateRBG_EngineOverride(t *testing.T) {
	// Engine flag overrides the mode's default engine
	rbg, metadata, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Engine:   "sglang",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	// Check returned metadata
	assert.Equal(t, "sglang", metadata.Engine)

	// Check metadata annotation
	var annotationMetadata llmmeta.RunMetadata
	err = json.Unmarshal([]byte(rbg.Annotations[llmmeta.RunCommandMetadataAnnotationKey]), &annotationMetadata)
	require.NoError(t, err)
	assert.Equal(t, "sglang", annotationMetadata.Engine)
}

func TestGenerateRBG_EnvVarInjection(t *testing.T) {
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Revision: "main",
		EnvVars:  []string{"MY_KEY=my_value"},
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	envMap := map[string]string{}
	for _, e := range podTemplate.Spec.Containers[0].Env {
		envMap[e.Name] = e.Value
	}
	assert.Equal(t, "my_value", envMap["MY_KEY"])
}

func TestGenerateRBG_InvalidEnvVar(t *testing.T) {
	_, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Revision: "main",
		EnvVars:  []string{"NOEQUALSSIGN"},
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid environment variable format")
}

func TestGenerateRBG_UnknownModel(t *testing.T) {
	// Unknown model should return error when no wildcard config exists
	_, _, err := generateRBG("my-svc", "unknown/unknown-model", "default", RunParams{
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no model config found for")
}

func TestGenerateRBG_UnknownEngine_Errors(t *testing.T) {
	_, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Engine:   "nonexistent-engine",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown engine type")
}

func TestGenerateRBG_AdditionalArgs(t *testing.T) {
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Revision: "main",
		ArgsList: []string{"--custom-flag", "value"},
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	args := podTemplate.Spec.Containers[0].Args
	assert.Contains(t, args, "--custom-flag")
	assert.Contains(t, args, "value")
}

func TestGenerateRBG_FallbackModelPath(t *testing.T) {
	// Without storage config, model path uses the /model/ fallback
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	args := podTemplate.Spec.Containers[0].Args
	var modelPathArg string
	for i, a := range args {
		if a == "--model" && i+1 < len(args) {
			modelPathArg = args[i+1]
			break
		}
	}
	assert.True(t, strings.HasPrefix(modelPathArg, "/model/"), "expected fallback model path, got: %s", modelPathArg)
}

// --- parseResources ---

func TestParseResources_Valid(t *testing.T) {
	res, err := parseResources([]string{"nvidia.com/gpu=1", "memory=16Gi", "cpu=4"})
	require.NoError(t, err)
	assert.Equal(t, 3, len(res))
	assert.True(t, res["nvidia.com/gpu"].Equal(resource.MustParse("1")))
	assert.True(t, res["memory"].Equal(resource.MustParse("16Gi")))
	assert.True(t, res["cpu"].Equal(resource.MustParse("4")))
}

func TestParseResources_InvalidFormat(t *testing.T) {
	_, err := parseResources([]string{"nvidia.com/gpu"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid resource format")
}

func TestParseResources_InvalidQuantity(t *testing.T) {
	_, err := parseResources([]string{"memory=notaquantity"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid resource quantity")
}

func TestParseResources_Empty(t *testing.T) {
	res, err := parseResources(nil)
	require.NoError(t, err)
	assert.Equal(t, 0, len(res))
}

// --- flag-only deployment (no model config) ---

func TestGenerateRBG_FlagOnly_WithEngine(t *testing.T) {
	// Unknown model + --engine specified should succeed
	rbg, metadata, err := generateRBG("my-custom", "custom/new-model", "default", RunParams{
		Engine:   "vllm",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "my-custom", rbg.Name)
	assert.Equal(t, "vllm", metadata.Engine)
	assert.Equal(t, "custom", metadata.Mode)
	assert.Equal(t, "custom/new-model", metadata.ModelID)
}

func TestGenerateRBG_FlagOnly_WithImageOverride(t *testing.T) {
	rbg, _, err := generateRBG("my-custom", "custom/new-model", "default", RunParams{
		Engine:   "vllm",
		Image:    "my-registry/vllm:custom",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	assert.Equal(t, "my-registry/vllm:custom", podTemplate.Spec.Containers[0].Image)
}

func TestGenerateRBG_FlagOnly_WithResources(t *testing.T) {
	rbg, _, err := generateRBG("my-custom", "custom/new-model", "default", RunParams{
		Engine:    "vllm",
		Resources: []string{"nvidia.com/gpu=2", "memory=16Gi"},
		Revision:  "main",
		Replicas:  1,
		DryRun:    true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	container := podTemplate.Spec.Containers[0]
	assert.True(t, container.Resources.Limits["nvidia.com/gpu"].Equal(resource.MustParse("2")))
	assert.True(t, container.Resources.Limits["memory"].Equal(resource.MustParse("16Gi")))
	assert.True(t, container.Resources.Requests["nvidia.com/gpu"].Equal(resource.MustParse("2")))
}

func TestGenerateRBG_FlagOnly_NoEngine_Errors(t *testing.T) {
	// Unknown model without --engine should fail
	_, _, err := generateRBG("my-custom", "custom/new-model", "default", RunParams{
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "--engine not specified")
}

// --- flag override on existing model config ---

func TestGenerateRBG_ImageOverride_ExistingModel(t *testing.T) {
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Image:    "custom-image:v2",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	assert.Equal(t, "custom-image:v2", podTemplate.Spec.Containers[0].Image)
}

func TestGenerateRBG_ResourceOverride_ExistingModel(t *testing.T) {
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Resources: []string{"nvidia.com/gpu=4"},
		Revision:  "main",
		Replicas:  1,
		DryRun:    true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	container := podTemplate.Spec.Containers[0]
	assert.True(t, container.Resources.Limits["nvidia.com/gpu"].Equal(resource.MustParse("4")))
}

func TestGenerateRBG_ShmSizeOverride(t *testing.T) {
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		ShmSize:  "32Gi",
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	// ShmSize should result in an emptyDir volume with sizeLimit
	foundShm := false
	for _, v := range podTemplate.Spec.Volumes {
		if v.Name == "shm" && v.EmptyDir != nil {
			foundShm = true
			expected := resource.MustParse("32Gi")
			assert.True(t, v.EmptyDir.SizeLimit.Equal(expected), "expected sizeLimit=32Gi")
		}
	}
	assert.True(t, foundShm, "expected shm volume with sizeLimit=32Gi")
}

// --- newRunCmd: new flags exist ---

func TestNewRunCmd_NewFlagDefaults(t *testing.T) {
	cf := genericclioptions.NewConfigFlags(true)
	cmd := newRunCmd(cf)

	imageFlag := cmd.Flags().Lookup("image")
	require.NotNil(t, imageFlag)
	assert.Equal(t, "", imageFlag.DefValue)

	resourceFlag := cmd.Flags().Lookup("resource")
	require.NotNil(t, resourceFlag)

	distFlag := cmd.Flags().Lookup("distributed-size")
	require.NotNil(t, distFlag)
	assert.Equal(t, "0", distFlag.DefValue)

	shmFlag := cmd.Flags().Lookup("shm-size")
	require.NotNil(t, shmFlag)
	assert.Equal(t, "", shmFlag.DefValue)

	tolFlag := cmd.Flags().Lookup("toleration")
	require.NotNil(t, tolFlag)
}

func TestParseToleration(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantKey   string
		wantValue string
		wantOp    corev1.TolerationOperator
		wantEff   corev1.TaintEffect
		wantErr   bool
	}{
		{
			name:    "key=value:NoSchedule (Equal)",
			input:   "gpu-type=a100:NoSchedule",
			wantKey: "gpu-type", wantValue: "a100",
			wantOp: corev1.TolerationOpEqual, wantEff: corev1.TaintEffectNoSchedule,
		},
		{
			name:    "key:NoSchedule (Exists)",
			input:   "node-role.alibabacloud.com/lingjun:NoSchedule",
			wantKey: "node-role.alibabacloud.com/lingjun",
			wantOp:  corev1.TolerationOpExists, wantEff: corev1.TaintEffectNoSchedule,
		},
		{
			name:    "key:NoExecute",
			input:   "dedicated:NoExecute",
			wantKey: "dedicated",
			wantOp:  corev1.TolerationOpExists, wantEff: corev1.TaintEffectNoExecute,
		},
		{
			name:    "key:PreferNoSchedule",
			input:   "special:PreferNoSchedule",
			wantKey: "special",
			wantOp:  corev1.TolerationOpExists, wantEff: corev1.TaintEffectPreferNoSchedule,
		},
		{
			name:    "key only (all effects)",
			input:   "my-key",
			wantKey: "my-key",
			wantOp:  corev1.TolerationOpExists, wantEff: "",
		},
		{
			name:      "key=value without effect",
			input:     "gpu-type=a100",
			wantKey:   "gpu-type",
			wantValue: "a100",
			wantOp:    corev1.TolerationOpEqual, wantEff: "",
		},
		{
			name:    "invalid effect",
			input:   "key:InvalidEffect",
			wantErr: true,
		},
		{
			name:    "empty string",
			input:   "",
			wantErr: true,
		},
		{
			name:    "empty key with effect",
			input:   ":NoSchedule",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseToleration(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantKey, got.Key)
			assert.Equal(t, tt.wantValue, got.Value)
			assert.Equal(t, tt.wantOp, got.Operator)
			assert.Equal(t, tt.wantEff, got.Effect)
		})
	}
}

func TestGenerateRBG_Toleration(t *testing.T) {
	rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
		Tolerations: []string{
			"node-role.alibabacloud.com/lingjun:NoSchedule",
			"gpu-type=a100:NoSchedule",
		},
		Revision: "main",
		Replicas: 1,
		DryRun:   true,
	}, nil, nil)
	require.NoError(t, err)

	podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
	require.Len(t, podTemplate.Spec.Tolerations, 2)

	tol0 := podTemplate.Spec.Tolerations[0]
	assert.Equal(t, "node-role.alibabacloud.com/lingjun", tol0.Key)
	assert.Equal(t, corev1.TolerationOpExists, tol0.Operator)
	assert.Equal(t, corev1.TaintEffectNoSchedule, tol0.Effect)

	tol1 := podTemplate.Spec.Tolerations[1]
	assert.Equal(t, "gpu-type", tol1.Key)
	assert.Equal(t, "a100", tol1.Value)
	assert.Equal(t, corev1.TolerationOpEqual, tol1.Operator)
	assert.Equal(t, corev1.TaintEffectNoSchedule, tol1.Effect)
}

// --- imagePullSecrets ---

func TestGenerateRBG_ImagePullSecrets(t *testing.T) {
	tests := []struct {
		name    string
		secrets []string
		want    []corev1.LocalObjectReference
		wantErr string
	}{
		{
			name:    "single secret",
			secrets: []string{"my-registry-secret"},
			want:    []corev1.LocalObjectReference{{Name: "my-registry-secret"}},
		},
		{
			name:    "multiple secrets",
			secrets: []string{"secret-a", "secret-b"},
			want:    []corev1.LocalObjectReference{{Name: "secret-a"}, {Name: "secret-b"}},
		},
		{
			name:    "no secrets",
			secrets: nil,
			want:    nil,
		},
		{
			name:    "empty string rejected",
			secrets: []string{""},
			wantErr: "image pull secret name must not be empty",
		},
		{
			name:    "whitespace-only rejected",
			secrets: []string{"  "},
			wantErr: "image pull secret name must not be empty",
		},
		{
			name:    "valid then empty rejected",
			secrets: []string{"good-secret", ""},
			wantErr: "image pull secret name must not be empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
				ImagePullSecrets: tt.secrets,
				Revision:         "main",
				Replicas:         1,
				DryRun:           true,
			}, nil, nil)

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)

			podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
			if tt.want == nil {
				assert.Empty(t, podTemplate.Spec.ImagePullSecrets)
			} else {
				assert.Equal(t, tt.want, podTemplate.Spec.ImagePullSecrets)
			}
		})
	}
}

// --- hostNetwork ---

func TestGenerateRBG_HostNetwork(t *testing.T) {
	tests := []struct {
		name        string
		hostNetwork bool
		want        bool
	}{
		{
			name:        "host network enabled",
			hostNetwork: true,
			want:        true,
		},
		{
			name:        "host network disabled (default)",
			hostNetwork: false,
			want:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
				HostNetwork: tt.hostNetwork,
				Revision:    "main",
				Replicas:    1,
				DryRun:      true,
			}, nil, nil)
			require.NoError(t, err)

			podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
			assert.Equal(t, tt.want, podTemplate.Spec.HostNetwork)
		})
	}
}

// --- nodeSelector ---

func TestGenerateRBG_NodeSelector(t *testing.T) {
	tests := []struct {
		name         string
		nodeSelector []string
		want         map[string]string
		wantErr      string
	}{
		{
			name:         "single label",
			nodeSelector: []string{"gpu-type=a100"},
			want:         map[string]string{"gpu-type": "a100"},
		},
		{
			name:         "multiple labels",
			nodeSelector: []string{"gpu-type=a100", "zone=us-east-1a"},
			want:         map[string]string{"gpu-type": "a100", "zone": "us-east-1a"},
		},
		{
			name:         "empty value allowed",
			nodeSelector: []string{"dedicated="},
			want:         map[string]string{"dedicated": ""},
		},
		{
			name:         "no node selector",
			nodeSelector: nil,
			want:         nil,
		},
		{
			name:         "invalid: missing equals",
			nodeSelector: []string{"noequalssign"},
			wantErr:      "invalid node-selector format",
		},
		{
			name:         "invalid: empty key",
			nodeSelector: []string{"=value"},
			wantErr:      "invalid node-selector format",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
				NodeSelector: tt.nodeSelector,
				Revision:     "main",
				Replicas:     1,
				DryRun:       true,
			}, nil, nil)
			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
				return
			}
			require.NoError(t, err)

			podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
			if tt.want == nil {
				assert.Empty(t, podTemplate.Spec.NodeSelector)
			} else {
				assert.Equal(t, tt.want, podTemplate.Spec.NodeSelector)
			}
		})
	}
}

// --- modelPrefetch ---

func TestGenerateRBG_ModelPrefetch(t *testing.T) {
	tests := []struct {
		name          string
		modelPrefetch bool
		modelPath     string
	}{
		{
			name:          "prefetch enabled with default path",
			modelPrefetch: true,
		},
		{
			name:          "prefetch enabled with custom path",
			modelPrefetch: true,
			modelPath:     "/custom/model/path",
		},
		{
			name:          "prefetch disabled",
			modelPrefetch: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rbg, _, err := generateRBG("my-svc", "Qwen/Qwen3.5-0.8B", "default", RunParams{
				ModelPrefetch: tt.modelPrefetch,
				ModelPath:     tt.modelPath,
				Revision:      "main",
				Replicas:      1,
				DryRun:        true,
			}, nil, nil)
			require.NoError(t, err)

			podTemplate := getPodTemplateFromPattern(&rbg.Spec.Roles[0].Pattern)
			container := podTemplate.Spec.Containers[0]

			if !tt.modelPrefetch {
				// No lifecycle hook should be set
				if container.Lifecycle != nil {
					assert.Nil(t, container.Lifecycle.PostStart)
				}
				// No __MODEL_PREFETCH_PATH env var
				for _, env := range container.Env {
					assert.NotEqual(t, "__MODEL_PREFETCH_PATH", env.Name)
				}
				return
			}

			// Lifecycle PostStart hook must be set
			require.NotNil(t, container.Lifecycle)
			require.NotNil(t, container.Lifecycle.PostStart)
			require.NotNil(t, container.Lifecycle.PostStart.Exec)
			assert.Equal(t, "/bin/sh", container.Lifecycle.PostStart.Exec.Command[0])
			assert.Equal(t, "-c", container.Lifecycle.PostStart.Exec.Command[1])

			// Verify the command includes timeout and the safe xargs idiom
			cmd := container.Lifecycle.PostStart.Exec.Command[2]
			assert.Contains(t, cmd, "timeout 600")
			assert.Contains(t, cmd, "$__MODEL_PREFETCH_PATH")
			assert.Contains(t, cmd, "xargs -0 -r -P 4 cat >/dev/null 2>&1")

			// __MODEL_PREFETCH_PATH env var must be set with correct path
			var prefetchEnv *corev1.EnvVar
			for i := range container.Env {
				if container.Env[i].Name == "__MODEL_PREFETCH_PATH" {
					prefetchEnv = &container.Env[i]
					break
				}
			}
			require.NotNil(t, prefetchEnv, "expected __MODEL_PREFETCH_PATH env var")
			if tt.modelPath != "" {
				assert.Equal(t, tt.modelPath, prefetchEnv.Value)
			} else {
				// Default path should start with /model/
				assert.True(t, strings.HasPrefix(prefetchEnv.Value, "/model/"),
					"expected default model path, got: %s", prefetchEnv.Value)
			}
		})
	}
}
