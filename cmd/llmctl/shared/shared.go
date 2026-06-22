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

package shared

import (
	"fmt"
	"os"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/yaml"

	workloadsv1alpha2 "sigs.k8s.io/rbgs/api/workloads/v1alpha2"
)

// SanitizeModelID sanitizes the model ID for use in resource names.
func SanitizeModelID(modelID string) string {
	result := strings.ReplaceAll(modelID, "/", "-")
	result = strings.ReplaceAll(result, ":", "-")
	result = strings.ReplaceAll(result, "_", "-")
	result = strings.ToLower(result)
	return result
}

// ToImagePullSecrets converts a string slice of secret names into
// Kubernetes LocalObjectReference values. It returns an error if any
// name is empty or whitespace-only.
func ToImagePullSecrets(names []string) ([]corev1.LocalObjectReference, error) {
	if len(names) == 0 {
		return nil, nil
	}
	refs := make([]corev1.LocalObjectReference, 0, len(names))
	for _, name := range names {
		if strings.TrimSpace(name) == "" {
			return nil, fmt.Errorf("image pull secret name must not be empty")
		}
		refs = append(refs, corev1.LocalObjectReference{Name: name})
	}
	return refs, nil
}

// PrintRBG prints a v1alpha2 RoleBasedGroup as YAML.
func PrintRBG(rbg *workloadsv1alpha2.RoleBasedGroup) error {
	out, err := yaml.Marshal(rbg)
	if err != nil {
		return fmt.Errorf("failed to marshal RoleBasedGroup: %w", err)
	}
	_, err = os.Stdout.Write(out)
	return err
}
