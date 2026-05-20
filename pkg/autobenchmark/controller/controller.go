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
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	sigsyaml "sigs.k8s.io/yaml"

	"sigs.k8s.io/rbgs/api/workloads/constants"
	"sigs.k8s.io/rbgs/api/workloads/v1alpha2"
	"sigs.k8s.io/rbgs/cli/pkg/autobenchmark/config"
	"sigs.k8s.io/rbgs/cli/pkg/autobenchmark/constant"
	"sigs.k8s.io/rbgs/cli/pkg/autobenchmark/evaluator"
	"sigs.k8s.io/rbgs/cli/pkg/autobenchmark/lifecycle"
	"sigs.k8s.io/rbgs/cli/pkg/autobenchmark/search"
	abtypes "sigs.k8s.io/rbgs/cli/pkg/autobenchmark/types"
)

// Controller orchestrates the auto-benchmark experiment loop.
type Controller struct {
	cfg       *config.AutoBenchmarkConfig
	client    client.Client
	clientset kubernetes.Interface
	namespace string

	builder *lifecycle.Builder
	manager *lifecycle.RBGManager
	eval    evaluator.Evaluator

	state     *StateManager
	reportDir string

	// parsed durations
	timeout         time.Duration
	rbgReadyTimeout time.Duration
	trialTimeout    time.Duration

	// sanitized experiment name for label values (DNS-1123 compliant)
	expNameLabel string
}

// NewController creates a Controller from config.
func NewController(
	cfg *config.AutoBenchmarkConfig,
	c client.Client,
	cs kubernetes.Interface,
	namespace string,
	stateDir string,
	reportDir string,
) (*Controller, error) {
	builder, err := lifecycle.NewBuilder(cfg.Backend)
	if err != nil {
		return nil, fmt.Errorf("creating builder: %w", err)
	}

	// Validate algorithm name early (don't store; Run creates per-template instances).
	if _, err := search.Get(cfg.Strategy.Algorithm); err != nil {
		return nil, fmt.Errorf("creating search algorithm: %w", err)
	}

	eval, err := evaluator.Get(cfg.Evaluator.Type)
	if err != nil {
		return nil, fmt.Errorf("creating evaluator: %w", err)
	}

	if err := eval.Init(cfg.Evaluator.Config); err != nil {
		return nil, fmt.Errorf("initializing evaluator: %w", err)
	}

	// Duration strings are validated during config parsing (setDefaults + Validate),
	// so errors here are effectively unreachable in normal flow.
	timeout, _ := time.ParseDuration(cfg.Strategy.Timeout)
	rbgReady, _ := time.ParseDuration(cfg.Execution.RBGReadyTimeout)
	trialTO, _ := time.ParseDuration(cfg.Execution.TrialTimeout)

	// Sanitize experiment name for use as Kubernetes label value
	expNameLabel := sanitizeLabelValue(cfg.Name)

	return &Controller{
		cfg:             cfg,
		client:          c,
		clientset:       cs,
		namespace:       namespace,
		builder:         builder,
		manager:         lifecycle.NewRBGManager(c, namespace),
		eval:            eval,
		state:           NewStateManager(stateDir),
		reportDir:       reportDir,
		timeout:         timeout,
		rbgReadyTimeout: rbgReady,
		trialTimeout:    trialTO,
		expNameLabel:    expNameLabel,
	}, nil
}

// Run executes the main orchestration loop.
func (ctrl *Controller) Run(ctx context.Context) error {
	logger := log.FromContext(ctx).WithValues("experiment", ctrl.cfg.Name)
	ctx = log.IntoContext(ctx, logger)

	// Apply overall timeout
	if ctrl.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, ctrl.timeout)
		defer cancel()
	}

	// Load or initialize experiment state
	expState, err := ctrl.loadOrInitState(ctx)
	if err != nil {
		return fmt.Errorf("initializing state: %w", err)
	}

	// Create algorithm instance once; Init will be called per template.
	algoInstance, err := search.Get(ctrl.cfg.Strategy.Algorithm)
	if err != nil {
		return err
	}
	// Close subprocess (if any) when the experiment finishes.
	if closer, ok := algoInstance.(interface{ Close() error }); ok {
		defer func() { _ = closer.Close() }()
	}

	// Template iteration loop
	iter := NewTemplateIterator(ctrl.cfg.Templates, expState.CurrentTemplateIdx)

	for iter.HasNext() {
		if err := ctx.Err(); err != nil {
			logger.Info("Experiment timeout reached, saving state")
			break
		}

		tmplRef := iter.Next()
		tmplIdx := iter.CurrentIndex() - 1 // index of current template
		logger.Info("Processing template", "index", tmplIdx, "template", tmplRef.Name)

		// Ensure template state exists
		for len(expState.Templates) <= tmplIdx {
			expState.Templates = append(expState.Templates, abtypes.TemplateState{
				Name: tmplRef.Name,
			})
		}
		ts := &expState.Templates[tmplIdx]

		if ts.Completed {
			logger.Info("Template already completed, skipping", "template", tmplRef.Name)
			continue
		}

		// (Re-)initialize algorithm for this template. The algorithm instance
		// is created once before the loop; Init reuses the subprocess.
		if err := algoInstance.Init(ctx, tmplRef.Name, ctrl.cfg.SearchSpace, ctrl.cfg.Strategy); err != nil {
			return fmt.Errorf("initializing algorithm for template %q: %w", tmplRef.Name, err)
		}
		if len(ts.AlgorithmState) > 0 {
			if err := algoInstance.UnmarshalState(ts.AlgorithmState); err != nil {
				logger.Info("Failed to restore algorithm state, continuing", "template", tmplRef.Name, "error", err.Error())
			}
		}

		// Load base RBG template
		baseRBG, err := lifecycle.LoadTemplate(tmplRef.Template)
		if err != nil {
			return fmt.Errorf("loading template %q: %w", tmplRef.Name, err)
		}

		// Trial loop for this template
		err = ctrl.runTrials(ctx, expState, ts, tmplIdx, baseRBG, algoInstance)
		if err != nil {
			logger.Error(err, "Error running trials", "template", tmplRef.Name)
		}

		// Mark template completed
		ts.Completed = true
		ts.BestTrial = SelectBest(ts.Trials)

		// Update global best
		if ts.BestTrial != nil {
			if expState.GlobalBest == nil || ts.BestTrial.Score > expState.GlobalBest.Score {
				expState.GlobalBest = ts.BestTrial
			}
		}

		expState.CurrentTemplateIdx = iter.CurrentIndex()
		if err := ctrl.state.Save(expState); err != nil {
			logger.Error(err, "Failed to save checkpoint")
		}
	}

	// Generate report
	endTime := time.Now()
	report := BuildReport(expState)
	if err := WriteReportJSON(ctrl.reportDir, report); err != nil {
		logger.Error(err, "Failed to write report", "format", "json")
	}
	if err := WriteReportYAML(ctrl.reportDir, report); err != nil {
		logger.Error(err, "Failed to write report", "format", "yaml")
	}
	// Write full result detail for UI consumption
	result := BuildResult(expState, ctrl.cfg, endTime)
	if err := WriteResultJSON(ctrl.reportDir, result); err != nil {
		logger.Error(err, "Failed to write result detail", "format", "json")
	}

	// Write best trial RBG YAML
	if expState.GlobalBest != nil {
		if err := ctrl.writeBestTrialYAML(expState.GlobalBest); err != nil {
			logger.Error(err, "Failed to write best trial YAML")
		}
	}

	logger.Info("Experiment completed", "summary", report.Summary)
	return nil
}

// runTrials executes the trial loop for a single template.
func (ctrl *Controller) runTrials(
	ctx context.Context,
	expState *abtypes.ExperimentState,
	ts *abtypes.TemplateState,
	tmplIdx int,
	baseRBG *v1alpha2.RoleBasedGroup,
	algo search.SearchAlgorithm,
) error {
	logger := log.FromContext(ctx).WithValues("template", ts.Name)

	for !algo.IsDone(ts.Trials) {
		if err := ctx.Err(); err != nil {
			return nil // timeout, graceful exit
		}

		trialIdx := len(ts.Trials)
		params, err := algo.SuggestNext(ts.Trials)
		if err != nil {
			logger.Error(err, "Algorithm suggest failed")
			break
		}

		logger.Info("Starting trial", "trialIndex", trialIdx, "params", params)

		result := ctrl.executeTrial(ctx, baseRBG, ts.Name, trialIdx, params)
		ts.Trials = append(ts.Trials, result)

		// Save algorithm state
		algoState, err := algo.MarshalState()
		if err == nil {
			ts.AlgorithmState = algoState
		}

		// Checkpoint after each trial
		expState.CurrentTemplateIdx = tmplIdx
		if err := ctrl.state.Save(expState); err != nil {
			logger.Error(err, "Failed to save checkpoint")
		}

		// Write incremental result detail for mid-flight visibility
		resultDetail := BuildResult(expState, ctrl.cfg, time.Time{})
		if err := WriteResultJSON(ctrl.reportDir, resultDetail); err != nil {
			logger.Error(err, "Failed to write result detail")
		}

		// Check early termination conditions
		if ctrl.cfg.Strategy.EarlyTermination != nil {
			if et := CheckEarlyTermination(ts.Trials, ctrl.cfg.Strategy.EarlyTermination); et.Terminated {
				logger.Info("Early termination triggered", "reason", et.Reason)
				ts.TerminationReason = et.Reason
				break
			}
		}
	}

	return nil
}

// executeTrial runs a single trial: deploy RBG -> wait ready -> run benchmark in-process -> collect results -> cleanup.
func (ctrl *Controller) executeTrial(
	ctx context.Context,
	baseRBG *v1alpha2.RoleBasedGroup,
	templateName string,
	trialIdx int,
	params abtypes.RoleParamSet,
) abtypes.TrialResult {
	logger := log.FromContext(ctx).WithValues("trialIndex", trialIdx)
	ctx = log.IntoContext(ctx, logger)

	start := time.Now()
	result := abtypes.TrialResult{
		TrialIndex:   trialIdx,
		TemplateName: templateName,
		Params:       params,
		StartTime:    start,
	}

	// Apply trial timeout early so it governs all trial phases
	// (creation, readiness, and benchmark execution).
	trialCtx := ctx
	if ctrl.trialTimeout > 0 {
		var cancel context.CancelFunc
		trialCtx, cancel = context.WithTimeout(ctx, ctrl.trialTimeout)
		defer cancel()
	}

	// Build trial RBG
	trialRBG, err := ctrl.builder.BuildTrial(baseRBG, trialIdx, params)
	if err != nil {
		result.Error = fmt.Sprintf("build trial: %v", err)
		result.EndTime = time.Now()
		result.Duration = abtypes.Duration(result.EndTime.Sub(start))
		return result
	}

	// Tag trial RBG with experiment label for scoped cleanup
	if trialRBG.Labels == nil {
		trialRBG.Labels = make(map[string]string)
	}
	trialRBG.Labels[constant.AutoBenchmarkLabelKey] = ctrl.expNameLabel
	if trialRBG.Annotations == nil {
		trialRBG.Annotations = make(map[string]string)
	}
	trialRBG.Annotations[constant.AutoBenchmarkOriginalNameAnnotationKey] = ctrl.cfg.Name

	trialName := trialRBG.Name
	rbgCreated := false

	defer func() {
		if !rbgCreated {
			return
		}
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cleanupCancel()
		err := wait.ExponentialBackoffWithContext(cleanupCtx, wait.Backoff{
			Duration: 5 * time.Second,
			Factor:   1,
			Steps:    3,
		}, func(ctx context.Context) (bool, error) {
			if delErr := ctrl.manager.Delete(ctx, trialName); delErr != nil {
				logger.Info("Failed to cleanup trial RBG, retrying", "rbgName", trialName, "error", delErr.Error())
				return false, nil
			}
			return true, nil
		})
		if err != nil {
			logger.Error(err, "Failed to cleanup trial RBG after retries", "rbgName", trialName)
		}
	}()

	// Create trial RBG
	if err := ctrl.manager.Create(trialCtx, trialRBG); err != nil {
		result.Error = fmt.Sprintf("create RBG: %v", err)
		result.EndTime = time.Now()
		result.Duration = abtypes.Duration(result.EndTime.Sub(start))
		ctrl.annotateTrialTimeout(&result, trialCtx)
		return result
	}
	rbgCreated = true

	// Wait for RBG to be fully ready (Pod Ready + inference endpoint serving).
	endpoint, err := ctrl.waitRBGFullyReady(trialCtx, trialRBG, trialName, ctrl.rbgReadyTimeout)
	if err != nil {
		result.Error = fmt.Sprintf("RBG not ready: %v", err)
		result.EndTime = time.Now()
		result.Duration = abtypes.Duration(result.EndTime.Sub(start))
		ctrl.annotateTrialTimeout(&result, trialCtx)

		scenario := ctrl.cfg.Scenario
		resultDir := filepath.Join(ctrl.reportDir, scenario.Name, templateName, fmt.Sprintf("trial-%d", trialIdx))
		ctrl.collectFailureLogs(trialName, resultDir, &result)
		return result
	}
	modelName := extractServedModelName(baseRBG, ctrl.cfg.Backend)

	// Result directory: {reportDir}/{scenario}/{templateName}/trial-{idx}
	scenario := ctrl.cfg.Scenario
	resultDir := filepath.Join(ctrl.reportDir, scenario.Name, templateName, fmt.Sprintf("trial-%d", trialIdx))

	// Snapshot pod restart counts before benchmark so we can detect mid-run crashes.
	preRunRestarts := ctrl.snapshotRestartCounts(trialName)

	// Run benchmark in-process
	evalCtx := evaluator.EvalContext{
		Endpoint:  endpoint,
		ModelName: modelName,
		Backend:   ctrl.cfg.Backend,
		Scenario:  scenario,
		OutputDir: resultDir,
	}

	if err := ctrl.eval.Run(trialCtx, evalCtx); err != nil {
		result.Error = fmt.Sprintf("benchmark failed: %v", err)
		result.EndTime = time.Now()
		result.Duration = abtypes.Duration(result.EndTime.Sub(start))
		ctrl.annotateTrialTimeout(&result, trialCtx)
		ctrl.collectBenchmarkFailureLogs(trialName, resultDir, preRunRestarts)
		return result
	}

	// Collect results from local output directory
	metrics, err := ctrl.eval.CollectResults(resultDir)
	if err != nil {
		result.Error = fmt.Sprintf("collecting results: %v", err)
		result.EndTime = time.Now()
		result.Duration = abtypes.Duration(result.EndTime.Sub(start))
		ctrl.annotateTrialTimeout(&result, trialCtx)
		return result
	}

	// Evaluate SLA
	result.Metrics = metrics
	result.Constraints, result.Score = EvaluateSLA(metrics, ctrl.cfg.Objectives)
	result.EndTime = time.Now()
	result.Duration = abtypes.Duration(result.EndTime.Sub(start))

	logger.Info("Trial completed", "feasible", result.IsSLAFeasible(), "score", result.Score, "constraints", result.Constraints)
	return result
}

// annotateTrialTimeout appends trial timeout context to an error result when
// the trial context's deadline has been exceeded.
func (ctrl *Controller) annotateTrialTimeout(result *abtypes.TrialResult, trialCtx context.Context) {
	if result.Error != "" && trialCtx.Err() == context.DeadlineExceeded {
		result.Error = fmt.Sprintf("%s (trial timeout after %s)", result.Error, ctrl.trialTimeout)
	}
}

// podRestartSnapshot maps "podName/containerName" to its RestartCount.
type podRestartSnapshot map[string]int32

// snapshotRestartCounts records the current RestartCount of every container
// across all pods belonging to the trial RBG. Returns nil on any error.
func (ctrl *Controller) snapshotRestartCounts(trialName string) podRestartSnapshot {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	podList, err := ctrl.clientset.CoreV1().Pods(ctrl.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labels.Set{constants.GroupNameLabelKey: trialName}.String(),
	})
	if err != nil || len(podList.Items) == 0 {
		return nil
	}

	snap := make(podRestartSnapshot)
	for i := range podList.Items {
		pod := &podList.Items[i]
		for _, cs := range append(pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses...) {
			snap[pod.Name+"/"+cs.Name] = cs.RestartCount
		}
	}
	return snap
}

// collectBenchmarkFailureLogs checks pod state after eval.Run failure and
// collects logs only when pods actually crashed during the benchmark:
//   - Pod in Failed phase → collect current logs (before RBG controller deletes it)
//   - Pod Running but RestartCount increased → collect previous container logs
//   - Neither → benchmark tool's own problem, skip
func (ctrl *Controller) collectBenchmarkFailureLogs(trialName string, resultDir string, preRunRestarts podRestartSnapshot) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	logger := log.FromContext(ctx).WithValues("trialName", trialName)

	podList, err := ctrl.clientset.CoreV1().Pods(ctrl.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labels.Set{constants.GroupNameLabelKey: trialName}.String(),
	})
	if err != nil || len(podList.Items) == 0 {
		return
	}

	var needLogs bool
	for i := range podList.Items {
		pod := &podList.Items[i]
		if pod.Status.Phase == corev1.PodFailed {
			needLogs = true
			break
		}
		if preRunRestarts != nil {
			for _, cs := range append(pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses...) {
				if cs.RestartCount > preRunRestarts[pod.Name+"/"+cs.Name] {
					needLogs = true
					break
				}
			}
		}
		if needLogs {
			break
		}
	}

	if !needLogs {
		return
	}

	logDir := filepath.Join(resultDir, "pod-logs")
	if err := os.MkdirAll(logDir, 0755); err != nil {
		logger.Info("Failed to create pod-logs directory", "error", err.Error())
		return
	}

	for i := range podList.Items {
		pod := &podList.Items[i]
		logPath := filepath.Join(logDir, pod.Name+".log")
		ctrl.writePodLogs(ctx, logPath, pod)
	}

	logger.Info("Collected benchmark failure logs", "logDir", logDir)
}

const failureLogTailLines int64 = 200

// collectFailureLogs fetches failure context from all pods belonging to the
// trial RBG. For Pending pods (never started), it appends a short summary
// directly to result.Error. For pods that did run, it writes the last N lines
// of container logs to {resultDir}/pod-logs/{podName}.log.
// Errors during collection are logged but never propagated.
func (ctrl *Controller) collectFailureLogs(trialName string, resultDir string, result *abtypes.TrialResult) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	logger := log.FromContext(ctx).WithValues("trialName", trialName)

	podList, err := ctrl.clientset.CoreV1().Pods(ctrl.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labels.Set{constants.GroupNameLabelKey: trialName}.String(),
	})
	if err != nil {
		logger.Info("Failed to list pods for failure log collection", "error", err.Error())
		return
	}

	if len(podList.Items) == 0 {
		logger.Info("No pods found for failure log collection")
		return
	}

	var pendingSummaries []string
	var logFileCount int

	for i := range podList.Items {
		pod := &podList.Items[i]

		if pod.Status.Phase == corev1.PodPending {
			if s := summarizePendingPod(pod); s != "" {
				pendingSummaries = append(pendingSummaries, s)
			}
			continue
		}

		logDir := filepath.Join(resultDir, "logs")
		if err := os.MkdirAll(logDir, 0755); err != nil {
			logger.Info("Failed to create logs directory", "error", err.Error())
			continue
		}
		logPath := filepath.Join(logDir, pod.Name+".log")
		ctrl.writePodLogs(ctx, logPath, pod)
		logFileCount++
	}

	if len(pendingSummaries) > 0 {
		result.Error = fmt.Sprintf("%s; pending pods: %s",
			result.Error, strings.Join(pendingSummaries, "; "))
	}

	if logFileCount > 0 {
		logger.Info("Collected failure logs", "logFiles", logFileCount, "dir", filepath.Join(resultDir, "logs"))
	}
}

// summarizePendingPod returns a one-line summary of why a pod is stuck in Pending.
func summarizePendingPod(pod *corev1.Pod) string {
	// Check container statuses (e.g. ImagePullBackOff, OOMKilled, Error)
	for _, cs := range append(pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses...) {
		if cs.State.Waiting != nil && cs.State.Waiting.Reason != "" {
			if cs.State.Waiting.Message != "" {
				return fmt.Sprintf("%s: %s (%s)", pod.Name, cs.State.Waiting.Reason, cs.State.Waiting.Message)
			}
			return fmt.Sprintf("%s: %s", pod.Name, cs.State.Waiting.Reason)
		}
		if cs.State.Terminated != nil && cs.State.Terminated.Reason != "" {
			return fmt.Sprintf("%s: %s (exitCode=%d)", pod.Name, cs.State.Terminated.Reason, cs.State.Terminated.ExitCode)
		}
	}
	// Check pod conditions (e.g. Unschedulable)
	for _, c := range pod.Status.Conditions {
		if c.Status == corev1.ConditionFalse && c.Reason != "" {
			if c.Message != "" {
				return fmt.Sprintf("%s: %s (%s)", pod.Name, c.Reason, c.Message)
			}
			return fmt.Sprintf("%s: %s", pod.Name, c.Reason)
		}
	}
	if pod.Status.Reason != "" {
		return fmt.Sprintf("%s: %s", pod.Name, pod.Status.Reason)
	}
	return fmt.Sprintf("%s: Pending", pod.Name)
}

// writePodLogs fetches the last N lines of logs from all containers in a pod.
// When a container has restarted (e.g. OOM crash), it also fetches the previous
// instance's logs which contain the actual crash output.
func (ctrl *Controller) writePodLogs(ctx context.Context, logPath string, pod *corev1.Pod) {
	var sb strings.Builder
	tailLines := failureLogTailLines

	restartCounts := make(map[string]int32)
	for _, cs := range append(pod.Status.InitContainerStatuses, pod.Status.ContainerStatuses...) {
		restartCounts[cs.Name] = cs.RestartCount
	}

	allContainers := make([]string, 0)
	for _, c := range pod.Spec.InitContainers {
		allContainers = append(allContainers, c.Name)
	}
	for _, c := range pod.Spec.Containers {
		allContainers = append(allContainers, c.Name)
	}

	for _, containerName := range allContainers {
		if restartCounts[containerName] > 0 {
			sb.WriteString(fmt.Sprintf("=== Container: %s (previous, restartCount=%d) ===\n", containerName, restartCounts[containerName]))
			ctrl.fetchContainerLogs(ctx, &sb, pod.Name, containerName, tailLines, true)
		}

		sb.WriteString(fmt.Sprintf("=== Container: %s ===\n", containerName))
		ctrl.fetchContainerLogs(ctx, &sb, pod.Name, containerName, tailLines, false)
	}

	_ = os.WriteFile(logPath, []byte(sb.String()), 0644)
}

func (ctrl *Controller) fetchContainerLogs(ctx context.Context, sb *strings.Builder, podName, containerName string, tailLines int64, previous bool) {
	req := ctrl.clientset.CoreV1().Pods(ctrl.namespace).GetLogs(podName, &corev1.PodLogOptions{
		Container: containerName,
		TailLines: &tailLines,
		Previous:  previous,
	})
	stream, err := req.Stream(ctx)
	if err != nil {
		sb.WriteString(fmt.Sprintf("(failed to get logs: %v)\n\n", err))
		return
	}
	logBytes, err := io.ReadAll(stream)
	_ = stream.Close()
	if err != nil {
		sb.WriteString(fmt.Sprintf("(failed to read logs: %v)\n\n", err))
		return
	}
	sb.Write(logBytes)
	sb.WriteString("\n\n")
}

// writeBestTrialYAML reconstructs the best trial's RBG and writes it as YAML.
func (ctrl *Controller) writeBestTrialYAML(best *abtypes.TrialResult) error {
	var tmplPath string
	for _, t := range ctrl.cfg.Templates {
		if t.Name == best.TemplateName {
			tmplPath = t.Template
			break
		}
	}
	if tmplPath == "" {
		return fmt.Errorf("template %q not found in config", best.TemplateName)
	}

	baseRBG, err := lifecycle.LoadTemplate(tmplPath)
	if err != nil {
		return fmt.Errorf("loading template %q: %w", best.TemplateName, err)
	}

	trialRBG, err := ctrl.builder.BuildTrial(baseRBG, best.TrialIndex, best.Params)
	if err != nil {
		return fmt.Errorf("building best trial RBG: %w", err)
	}

	data, err := sigsyaml.Marshal(trialRBG)
	if err != nil {
		return fmt.Errorf("marshaling best trial RBG: %w", err)
	}

	outPath := filepath.Join(ctrl.reportDir, "best_trial.yaml")
	if err := os.WriteFile(outPath, data, 0644); err != nil {
		return fmt.Errorf("writing best trial YAML: %w", err)
	}
	return nil
}

// resolveEndpoint determines the inference endpoint for a trial RBG.
func (ctrl *Controller) resolveEndpoint(trialRBG *v1alpha2.RoleBasedGroup) string {
	for _, role := range trialRBG.Spec.Roles {
		port := ctrl.resolveRolePort(&role)
		if port <= 0 {
			continue
		}
		// For leader-worker pattern, route directly to the leader pod via headless service DNS.
		// The RBG controller only creates headless services; using service-level DNS may resolve
		// to worker pods, which do not serve inference requests.
		if role.LeaderWorkerPattern != nil {
			return lifecycle.GetLeaderPodEndpoint(trialRBG, &role, ctrl.namespace, port)
		}
		return lifecycle.GetServiceEndpoint(trialRBG, &role, ctrl.namespace, port)
	}
	// Last resort fallback: assume a default worker role at port 8000.
	return lifecycle.GetServiceEndpoint(trialRBG, &v1alpha2.RoleSpec{Name: "worker"}, ctrl.namespace, 8000)
}

// resolveRolePort extracts the inference port for a role.
// Priority: args --port > ServicePorts > container ports > engine default.
func (ctrl *Controller) resolveRolePort(role *v1alpha2.RoleSpec) int {
	podSpec := getRolePodSpec(role)
	if podSpec != nil && len(podSpec.Containers) > 0 {
		// 1. Check container args for explicit --port.
		for _, c := range podSpec.Containers {
			for i, arg := range c.Args {
				if arg == "--port" && i+1 < len(c.Args) {
					if p, err := strconv.Atoi(c.Args[i+1]); err == nil && p > 0 {
						return p
					}
				}
			}
		}
		// 2. Check container ports.
		for _, c := range podSpec.Containers {
			for _, p := range c.Ports {
				if p.ContainerPort > 0 {
					return int(p.ContainerPort)
				}
			}
		}
	}
	// 3. Check ServicePorts.
	if len(role.ServicePorts) > 0 {
		return int(role.ServicePorts[0].Port)
	}
	// 4. Fall back to engine default.
	return defaultEnginePort(ctrl.cfg.Backend)
}

func defaultEnginePort(backend string) int {
	switch backend {
	case "sglang":
		return 30000
	case "vllm":
		return 8000
	default:
		return 8000
	}
}

// extractServedModelName reads --served-model-name from the base template's container args.
// Falls back to the RBG metadata name if the flag is not found.
func extractServedModelName(rbg *v1alpha2.RoleBasedGroup, backend string) string {
	flag := "--served-model-name"
	if backend == "vllm" {
		flag = "--served-model-name"
	}
	for _, role := range rbg.Spec.Roles {
		podSpec := getRolePodSpec(&role)
		if podSpec == nil {
			continue
		}
		for _, c := range podSpec.Containers {
			allArgs := append(c.Command, c.Args...)
			for i, arg := range allArgs {
				if arg == flag && i+1 < len(allArgs) {
					return allArgs[i+1]
				}
			}
		}
	}
	return rbg.Name
}

// getRolePodSpec extracts the PodSpec from a RoleSpec regardless of pattern type.
func getRolePodSpec(role *v1alpha2.RoleSpec) *corev1.PodSpec {
	if sp := role.StandalonePattern; sp != nil {
		if sp.Template != nil {
			return &sp.Template.Spec
		}
	}
	if lw := role.LeaderWorkerPattern; lw != nil {
		if lw.Template != nil {
			return &lw.Template.Spec
		}
	}
	return nil
}

// loadOrInitState loads checkpoint or creates new state.
func (ctrl *Controller) loadOrInitState(ctx context.Context) (*abtypes.ExperimentState, error) {
	existing, err := ctrl.state.Load()
	if err != nil {
		return nil, err
	}
	if existing != nil {
		logger := log.FromContext(ctx)
		logger.Info("Resuming experiment from checkpoint", "templateIndex", existing.CurrentTemplateIdx)
		return existing, nil
	}

	return &abtypes.ExperimentState{
		ExperimentID: ctrl.cfg.Name,
		StartTime:    time.Now(),
	}, nil
}

// waitRBGFullyReady waits for both the RBG to report Ready=True and the
// inference endpoint to respond with HTTP 200, sharing a single timeout.
// A bool tracks whether the RBG ready phase is complete so that subsequent
// polls only hit the endpoint (reducing unnecessary API calls).
func (ctrl *Controller) waitRBGFullyReady(
	ctx context.Context,
	trialRBG *v1alpha2.RoleBasedGroup,
	trialName string,
	timeout time.Duration,
) (endpoint string, err error) {
	logger := log.FromContext(ctx)
	rbgReady := false
	httpClient := &http.Client{Timeout: 5 * time.Second}

	err = wait.PollUntilContextTimeout(ctx, 10*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		if !rbgReady {
			rbg, err := ctrl.manager.Get(ctx, trialName)
			if err != nil {
				logger.V(2).Info("RBG not found yet", "error", err.Error())
				return false, nil // retry on transient errors
			}
			if !isRBGReady(rbg) {
				return false, nil
			}
			logger.Info("RBG is ready, waiting for inference endpoint", "rbgName", trialName)
			rbgReady = true
		}

		endpoint = ctrl.resolveEndpoint(trialRBG)
		healthURL := endpoint + "/health"
		resp, err := httpClient.Get(healthURL)
		if err != nil {
			logger.V(2).Info("Endpoint not ready yet", "error", err.Error())
			return false, nil // retry
		}
		defer func() { _ = resp.Body.Close() }()

		if resp.StatusCode == http.StatusOK {
			logger.Info("Inference endpoint is ready", "healthURL", healthURL)
			return true, nil
		}
		logger.V(2).Info("Endpoint returned non-OK status", "statusCode", resp.StatusCode)
		return false, nil
	})
	return endpoint, err
}

// isRBGReady checks if the RBG has a Ready=True condition.
func isRBGReady(rbg *v1alpha2.RoleBasedGroup) bool {
	for _, c := range rbg.Status.Conditions {
		if c.Type == string(v1alpha2.RoleBasedGroupReady) && c.Status == "True" {
			return true
		}
	}
	return false
}

// sanitizeLabelValue ensures a string is valid for use as a Kubernetes label value.
// Label values must be 63 characters or less and contain only alphanumeric characters,
// '-', '_', or '.'. This function truncates and sanitizes as needed.
func sanitizeLabelValue(name string) string {
	if len(name) > 63 {
		name = name[:63]
	}
	// Replace invalid characters with '-'
	invalidChars := regexp.MustCompile(`[^a-zA-Z0-9._-]`)
	name = invalidChars.ReplaceAllString(name, "-")
	name = strings.Trim(name, "-")
	if name == "" {
		name = "default"
	}
	return name
}
