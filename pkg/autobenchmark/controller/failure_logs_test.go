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
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIdentifyFailedPods(t *testing.T) {
	tests := []struct {
		name           string
		pods           []corev1.Pod
		preRunRestarts podRestartSnapshot
		wantNames      []string
	}{
		{
			name:           "no pods returns nil",
			pods:           nil,
			preRunRestarts: nil,
			wantNames:      nil,
		},
		{
			name: "pod in Failed phase is collected",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "failed-pod"},
					Status:     corev1.PodStatus{Phase: corev1.PodFailed},
				},
			},
			preRunRestarts: nil,
			wantNames:      []string{"failed-pod"},
		},
		{
			name: "pod in Running phase with no restart increase is skipped",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "healthy-pod"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "main", RestartCount: 2},
						},
					},
				},
			},
			preRunRestarts: podRestartSnapshot{"healthy-pod/main": 2},
			wantNames:      nil,
		},
		{
			name: "container restart count increased is collected",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "oom-pod"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "vllm", RestartCount: 3},
						},
					},
				},
			},
			preRunRestarts: podRestartSnapshot{"oom-pod/vllm": 2},
			wantNames:      []string{"oom-pod"},
		},
		{
			name: "init container restart count increased is collected",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "init-crash-pod"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						InitContainerStatuses: []corev1.ContainerStatus{
							{Name: "init", RestartCount: 1},
						},
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "main", RestartCount: 0},
						},
					},
				},
			},
			preRunRestarts: podRestartSnapshot{
				"init-crash-pod/init": 0,
				"init-crash-pod/main": 0,
			},
			wantNames: []string{"init-crash-pod"},
		},
		{
			name: "nil snapshot skips restart check but still detects PodFailed",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "crashed-pod"},
					Status:     corev1.PodStatus{Phase: corev1.PodFailed},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "running-restarted"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "main", RestartCount: 5},
						},
					},
				},
			},
			preRunRestarts: nil,
			wantNames:      []string{"crashed-pod"},
		},
		{
			name: "mixed: Failed pod and restarted pod both collected",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-failed"},
					Status:     corev1.PodStatus{Phase: corev1.PodFailed},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-oom"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "engine", RestartCount: 1},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod-healthy"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "engine", RestartCount: 0},
						},
					},
				},
			},
			preRunRestarts: podRestartSnapshot{
				"pod-oom/engine":     0,
				"pod-healthy/engine": 0,
			},
			wantNames: []string{"pod-failed", "pod-oom"},
		},
		{
			name: "new container not in snapshot is collected (restart from 0 to 1)",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "new-container-pod"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "sidecar", RestartCount: 1},
						},
					},
				},
			},
			preRunRestarts: podRestartSnapshot{},
			wantNames:      []string{"new-container-pod"},
		},
		{
			name: "Succeeded pod is not collected",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "done-pod"},
					Status:     corev1.PodStatus{Phase: corev1.PodSucceeded},
				},
			},
			preRunRestarts: podRestartSnapshot{},
			wantNames:      nil,
		},
		{
			name: "multiple containers - only one restarted triggers collection",
			pods: []corev1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "multi-container-pod"},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
						ContainerStatuses: []corev1.ContainerStatus{
							{Name: "engine", RestartCount: 0},
							{Name: "router", RestartCount: 2},
						},
					},
				},
			},
			preRunRestarts: podRestartSnapshot{
				"multi-container-pod/engine": 0,
				"multi-container-pod/router": 1,
			},
			wantNames: []string{"multi-container-pod"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := identifyFailedPods(tt.pods, tt.preRunRestarts)
			var gotNames []string
			for _, p := range got {
				gotNames = append(gotNames, p.Name)
			}
			assert.Equal(t, tt.wantNames, gotNames)
		})
	}
}

func TestSummarizePendingPod(t *testing.T) {
	tests := []struct {
		name string
		pod  *corev1.Pod
		want string
	}{
		{
			name: "container Waiting with message",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason:  "ImagePullBackOff",
									Message: "back-off pulling image",
								},
							},
						},
					},
				},
			},
			want: "pod-1: ImagePullBackOff (back-off pulling image)",
		},
		{
			name: "container Waiting without message",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason: "ContainerCreating",
								},
							},
						},
					},
				},
			},
			want: "pod-1: ContainerCreating",
		},
		{
			name: "container Terminated",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					ContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Terminated: &corev1.ContainerStateTerminated{
									Reason:   "OOMKilled",
									ExitCode: 137,
								},
							},
						},
					},
				},
			},
			want: "pod-1: OOMKilled (exitCode=137)",
		},
		{
			name: "init container Waiting",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					InitContainerStatuses: []corev1.ContainerStatus{
						{
							State: corev1.ContainerState{
								Waiting: &corev1.ContainerStateWaiting{
									Reason: "ImagePullBackOff",
								},
							},
						},
					},
				},
			},
			want: "pod-1: ImagePullBackOff",
		},
		{
			name: "pod condition false with message",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:    corev1.PodScheduled,
							Status:  corev1.ConditionFalse,
							Reason:  "Unschedulable",
							Message: "0/3 nodes are available: insufficient nvidia.com/gpu",
						},
					},
				},
			},
			want: "pod-1: Unschedulable (0/3 nodes are available: insufficient nvidia.com/gpu)",
		},
		{
			name: "pod condition false without message",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodScheduled,
							Status: corev1.ConditionFalse,
							Reason: "Unschedulable",
						},
					},
				},
			},
			want: "pod-1: Unschedulable",
		},
		{
			name: "pod status reason",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status: corev1.PodStatus{
					Reason: "EvictionByEvictionAPI",
				},
			},
			want: "pod-1: EvictionByEvictionAPI",
		},
		{
			name: "plain pending - no status info",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-1"},
				Status:     corev1.PodStatus{},
			},
			want: "pod-1: Pending",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, summarizePendingPod(tt.pod))
		})
	}
}
