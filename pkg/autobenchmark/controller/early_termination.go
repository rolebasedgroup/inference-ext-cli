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

package controller

import (
	"fmt"

	"sigs.k8s.io/rbgs/cli/pkg/autobenchmark/config"
	abtypes "sigs.k8s.io/rbgs/cli/pkg/autobenchmark/types"
)

// EarlyTerminationResult holds the outcome of an early termination check.
type EarlyTerminationResult struct {
	Terminated bool
	Reason     string
}

// CheckEarlyTermination evaluates whether the trial loop should be stopped
// based on the configured early termination conditions.
func CheckEarlyTermination(trials []abtypes.TrialResult, spec *config.EarlyTerminationSpec) EarlyTerminationResult {
	if spec == nil || len(trials) == 0 {
		return EarlyTerminationResult{}
	}

	// Universal guard: skip all checks until MinTrials have completed.
	if spec.MinTrials > 0 && len(trials) < spec.MinTrials {
		return EarlyTerminationResult{}
	}

	// Check consecutive SLA failures (from the tail of the trial list).
	if spec.MaxConsecutiveSLAFailures > 0 {
		consecutive := 0
		for i := len(trials) - 1; i >= 0; i-- {
			if !trials[i].IsSLAFeasible() {
				consecutive++
			} else {
				break
			}
		}
		if consecutive >= spec.MaxConsecutiveSLAFailures {
			return EarlyTerminationResult{
				Terminated: true,
				Reason: fmt.Sprintf("consecutive SLA failures reached limit: %d/%d",
					consecutive, spec.MaxConsecutiveSLAFailures),
			}
		}
	}

	// Check overall SLA failure rate.
	if spec.MaxSLAFailureRate > 0 {
		failures := 0
		for i := range trials {
			if !trials[i].IsSLAFeasible() {
				failures++
			}
		}
		rate := float64(failures) / float64(len(trials))
		if rate > spec.MaxSLAFailureRate {
			return EarlyTerminationResult{
				Terminated: true,
				Reason: fmt.Sprintf("SLA failure rate exceeded limit: %.2f > %.2f (%d/%d trials failed)",
					rate, spec.MaxSLAFailureRate, failures, len(trials)),
			}
		}
	}

	return EarlyTerminationResult{}
}
