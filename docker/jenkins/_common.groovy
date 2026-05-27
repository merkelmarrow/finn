// Shared utilities for docker/jenkins/Jenkinsfile and docker/jenkins/Jenkinsfile_HW.
// Loaded via load() in each Jenkinsfile once the SCM checkout has dropped the
// repo onto the active node, then accessed through thin top-level wrappers in
// the calling Jenkinsfile so existing call sites read the same as before.
//
// Adding a helper here keeps both pipelines from drifting. Helpers that are
// genuinely pipeline-specific (e.g. PARALLEL_SHARDS, HW_SHARDS, the various
// printLsfSummary helpers) stay in their owning Jenkinsfile.

boolean paramBool(String name) {
  def v = params.get(name)
  if (v == null) { return false }
  if (v instanceof Boolean) { return v }
  return v.toString().toBoolean()
}

String paramString(String name) {
  def v = params.get(name)
  return v == null ? '' : v.toString()
}

String shellQuote(String s) {
  return "'" + (s ?: '').replace("'", "'\"'\"'") + "'"
}

// Hints run-docker.sh with FINN_DOCKER_PREBUILT=1 when a shared image is
// configured so non-builder agents docker-load from NFS instead of rebuilding.
// Falls through to a local image build in local-fallback mode.
void runDockerCommand(String command) {
  if (env.FINN_DOCKER_SHARED_IMAGE_DIR) {
    withEnv(['FINN_DOCKER_PREBUILT=1']) {
      sh command
    }
  } else {
    sh command
  }
}

void unstashIfPresent(String stashName) {
  try {
    unstash stashName
  } catch (Exception ignored) {
    echo "No stash '${stashName}' (stage skipped or failed before publishing)"
  }
}

// SW form: stashes the full per-shard report sidecar set (xml, html,
// timings.json, shardmap.*, stagemap). allowEmpty=true because not every
// shard produces every sidecar.
void safeStashShardReport(String stashName) {
  catchError(buildResult: null, stageResult: null,
             message: "safeStashReport(${stashName}) failed, aggregation may be partial") {
    stash name: stashName,
          includes: "${stashName}.xml,${stashName}.html,${stashName}.timings.json," +
                    "${stashName}.shardmap.txt,${stashName}.shardmap.json,${stashName}.stagemap",
          allowEmpty: true
  }
}

// HW form: stash only when the JUnit XML exists, with a different filename
// pattern from the stashName (HW tests file their reports under
// "${testType}_hw_${board}" while the stash is keyed on "${testType}_${board}").
void safeStashHwReport(String stashName, String fileBase) {
  catchError(buildResult: null, stageResult: null,
             message: "safeStashReport(${stashName}) failed, aggregation may be partial") {
    if (fileExists("${fileBase}.xml")) {
      stash name: stashName,
            includes: "${fileBase}.xml,${fileBase}.html",
            allowEmpty: false
    }
  }
}

// SW form: rm with tolerance, hard-fail on root-owned residue, then mkdir.
// Pre-create as the unprivileged user so docker's -v doesn't bind as root.
void cleanPreviousBuildFiles(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String q = shellQuote(buildDir)
  sh """
    rm -rf ${q} 2>/dev/null || true
    if [ -d ${q} ]; then
      echo "cleanPreviousBuildFiles: ${q} still exists after rm. Likely root-owned residue. Ask an admin to 'sudo rm -rf' the directory on this agent."
      ls -la ${q} | head -40
      exit 1
    fi
    mkdir -p ${q}
  """
}

// HW form: rm with optional sudo when HW credentials are bound, and an
// optional sibling sweep (e.g. cleaning KV260_SOM/ also sweeps KV260_SOM.zip).
// Targets are enumerated explicitly so the next maintainer does not have to
// derive the behaviour from glob-expansion semantics.
void cleanPreviousBuildFilesHw(String buildDir, boolean includeSiblings) {
  if (!buildDir || buildDir.empty) { return }
  String prefix = env.USER_CREDENTIALS ? 'echo "$USER_CREDENTIALS_PSW" | sudo -S ' : ''
  String targets = includeSiblings
      ? "${shellQuote(buildDir)} ${shellQuote(buildDir + '.zip')}"
      : shellQuote(buildDir)
  sh "${prefix}rm -rf ${targets}"
}

// FINN_CI_NFS_ROOT is the only env var operators set. Every shared tree
// derives from it. Returning '' means "no NFS available" and callers
// MUST handle the fallback (skip cache mounts, skip image publish, skip
// artifact handoff, skip persistent timing master). Local fallback is a
// degraded but functional mode for software-only debug runs.
String finnCiNfsRoot() { return (env.FINN_CI_NFS_ROOT ?: '').trim() }

// Shared "anchor a subtree under FINN_CI_NFS_ROOT" helper so every callsite
// shares the same '' fallback semantics. Empty suffix segments collapse out
// so finnAgentCachesDir('') (missing NODE_NAME) returns '' rather than a
// dangling /agent_caches/ path.
String finnSubdir(String... segments) {
  String r = finnCiNfsRoot()
  if (!r) { return '' }
  for (int i = 0; i < segments.length; i++) {
    if (!segments[i]) { return '' }
  }
  return ([r] + (segments as List)).join('/')
}

String finnAgentCachesDir(String node)    { return finnSubdir('agent_caches', node) }
String finnDockerImagesRoot()             { return finnSubdir('docker_images') }
String finnDockerImagesDir(String jobKey) { return finnSubdir('docker_images', jobKey) }
String finnArtifactsRoot()                { return finnSubdir('artifacts') }
String finnCiStateDir(String jobKey)      { return finnSubdir('_ci_state', jobKey) }

return this
