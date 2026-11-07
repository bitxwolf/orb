import { useState, useReducer, useEffect, useCallback } from 'react';
import { Box, Text, useApp } from 'ink';

import type { HardwareInfo } from './types/hardware.js';
import type { ModelInfo } from './types/model.js';
import type { DatasetMeta } from './types/dataset.js';
import type { AutoConfigResult, EngineConfig } from './types/config.js';
import type {
  ProgressMessage,
} from './types/ipc.js';

import { detectHardware } from './core/hardware.js';
import { computeAutoConfig } from './core/auto-config.js';
import { areDependenciesReady, setupDependencies } from './core/dependencies.js';
import { spawnTrainingEngine, createDefaultCallbacks } from './core/ipc.js';
import { generateModelName, writeModelfile, importToOllama } from './core/ollama.js';
import { generateRunDir, ensureDir } from './utils/fs.js';
import { toPlainLanguageError } from './utils/errors.js';

import { Welcome } from './components/Welcome.js';
import { DatasetStep } from './components/DatasetStep.js';
import { ModelStep } from './components/ModelStep.js';
import { ConfigStep } from './components/ConfigStep.js';
import { TrainingStep } from './components/TrainingStep.js';
import { ExportStep, createDefaultExportStages } from './components/ExportStep.js';
import { DoneStep } from './components/DoneStep.js';
import { ErrorBox } from './components/ui/ErrorBox.js';
import { Spinner } from './components/ui/Spinner.js';
import { TabHeader } from './components/ui/TabHeader.js';
import { Mascot } from './components/ui/Mascot.js';
import { INK, glyph } from './utils/colors.js';

// ─── Wizard State ────────────────────────────────────────────────────────────

type WizardStep =
  | 'initializing'   // Detecting hardware
  | 'welcome'        // Show banner + hardware summary
  | 'dataset'        // Dataset selection + preview
  | 'model'          // Model ID input + validation
  | 'config'         // Auto-config review + confirmation
  | 'dependencies'   // Silent dep install (if needed)
  | 'training'       // Live training progress
  | 'export'         // GGUF conversion
  | 'done'           // Success!
  | 'error';         // Error display with guidance

interface WizardState {
  step: WizardStep;
  hardware: HardwareInfo | null;
  datasetMeta: DatasetMeta | null;
  model: ModelInfo | null;
  hfToken?: string;
  config: AutoConfigResult | null;
  // Training progress
  trainingProgress: ProgressMessage | null;
  trainingStage: string;
  lastCheckpoint: string | null;
  initialLoss: number | null;
  // Export progress
  exportStages: ReturnType<typeof createDefaultExportStages>;
  // Results
  adapterPath: string | null;
  ggufPath: string | null;
  modelfilePath: string | null;
  ollamaModelName: string | null;
  ollamaImported: boolean;
  ggufFileSize: string | null;
  elapsedSeconds: number | null;
  // Error
  error: { message: string; guidance?: string } | null;
}

type WizardAction =
  | { type: 'SET_STEP'; step: WizardStep }
  | { type: 'SET_HARDWARE'; hardware: HardwareInfo }
  | { type: 'SET_DATASET'; meta: DatasetMeta }
  | { type: 'SET_MODEL'; model: ModelInfo; hfToken?: string }
  | { type: 'SET_CONFIG'; config: AutoConfigResult }
  | { type: 'SET_TRAINING_PROGRESS'; progress: ProgressMessage }
  | { type: 'SET_TRAINING_STAGE'; stage: string }
  | { type: 'SET_CHECKPOINT'; path: string }
  | { type: 'SET_INITIAL_LOSS'; loss: number }
  | { type: 'SET_EXPORT_STAGE'; id: string; status: 'pending' | 'active' | 'complete' | 'error'; errorMessage?: string }
  | { type: 'SET_RESULTS'; adapterPath?: string; ggufPath?: string; modelfilePath?: string; ollamaModelName?: string; ollamaImported?: boolean; ggufFileSize?: string; elapsedSeconds?: number }
  | { type: 'SET_ERROR'; message: string; guidance?: string };

function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case 'SET_STEP':
      return { ...state, step: action.step };
    case 'SET_HARDWARE':
      return { ...state, hardware: action.hardware };
    case 'SET_DATASET':
      return { ...state, datasetMeta: action.meta };
    case 'SET_MODEL':
      return { ...state, model: action.model, hfToken: action.hfToken };
    case 'SET_CONFIG':
      return { ...state, config: action.config };
    case 'SET_TRAINING_PROGRESS':
      return { ...state, trainingProgress: action.progress };
    case 'SET_TRAINING_STAGE':
      return { ...state, trainingStage: action.stage };
    case 'SET_CHECKPOINT':
      return { ...state, lastCheckpoint: action.path };
    case 'SET_INITIAL_LOSS':
      return { ...state, initialLoss: action.loss };
    case 'SET_EXPORT_STAGE': {
      const stages = state.exportStages.map((s) =>
        s.id === action.id ? { ...s, status: action.status, errorMessage: action.errorMessage } : s
      );
      return { ...state, exportStages: stages };
    }
    case 'SET_RESULTS':
      return {
        ...state,
        adapterPath: action.adapterPath ?? state.adapterPath,
        ggufPath: action.ggufPath ?? state.ggufPath,
        modelfilePath: action.modelfilePath ?? state.modelfilePath,
        ollamaModelName: action.ollamaModelName ?? state.ollamaModelName,
        ollamaImported: action.ollamaImported ?? state.ollamaImported,
        ggufFileSize: action.ggufFileSize ?? state.ggufFileSize,
        elapsedSeconds: action.elapsedSeconds ?? state.elapsedSeconds,
      };
    case 'SET_ERROR':
      return { ...state, step: 'error', error: { message: action.message, guidance: action.guidance } };
    default:
      return state;
  }
}

const initialState: WizardState = {
  step: 'initializing',
  hardware: null,
  datasetMeta: null,
  model: null,
  config: null,
  trainingProgress: null,
  trainingStage: '',
  lastCheckpoint: null,
  initialLoss: null,
  exportStages: createDefaultExportStages(),
  adapterPath: null,
  ggufPath: null,
  modelfilePath: null,
  ollamaModelName: null,
  ollamaImported: false,
  ggufFileSize: null,
  elapsedSeconds: null,
  error: null,
};

// ─── Props ───────────────────────────────────────────────────────────────────

interface AppProps {
  resumeId?: string;
  debug?: boolean;
}

// ─── Main App Component ──────────────────────────────────────────────────────

export function App({ debug = false }: AppProps) {
  const { exit } = useApp();
  const [state, dispatch] = useReducer(wizardReducer, initialState);
  const [depStatus, setDepStatus] = useState('');

  // ── Initialize: detect hardware ──
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const hw = await detectHardware();
        if (!cancelled) {
          dispatch({ type: 'SET_HARDWARE', hardware: hw });
          // Start at model instead of welcome/dataset
          dispatch({ type: 'SET_STEP', step: 'model' });
        }
      } catch (err) {
        if (!cancelled) {
          const e = toPlainLanguageError(err);
          dispatch({ type: 'SET_ERROR', message: e.userMessage, guidance: e.guidance });
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Start training ──
  const startTraining = useCallback(async () => {
    if (!state.hardware || !state.model || !state.datasetMeta || !state.config) return;

    dispatch({ type: 'SET_STEP', step: 'training' });

    const runDir = generateRunDir(state.model.id);
    const checkpointDir = `${runDir}/checkpoints`;
    try {
      await ensureDir(runDir);
      await ensureDir(checkpointDir);
    } catch (err) {
      const e = toPlainLanguageError(err);
      dispatch({
        type: 'SET_ERROR',
        message: e.userMessage,
        guidance: debug && e.technicalDetail
          ? [e.guidance, e.technicalDetail].filter(Boolean).join('\n')
          : e.guidance,
      });
      return;
    }

    const modelName = generateModelName(state.model.id, state.datasetMeta.path);
    const ggufOutputPath = `${runDir}/${modelName}.gguf`;

    const engineConfig: EngineConfig = {
      backend: state.hardware.backend as 'cuda' | 'mlx',
      model_id: state.model.id,
      hf_token: state.hfToken,
      dataset_path: state.datasetMeta.normalizedPath,
      output_dir: runDir,
      checkpoint_dir: checkpointDir,
      seq_length: state.config.seqLength,
      lora: {
        rank: state.config.lora.rank,
        alpha: state.config.lora.alpha,
        dropout: state.config.lora.dropout,
        target_modules: state.config.lora.targetModules,
      },
      training: {
        batch_size: state.config.training.batchSize,
        gradient_accumulation_steps: state.config.training.gradientAccumulationSteps,
        learning_rate: state.config.training.learningRate,
        epochs: state.config.training.epochs,
        warmup_ratio: state.config.training.warmupRatio,
        max_steps: state.config.training.maxSteps,
        save_steps_ratio: state.config.training.saveStepsRatio,
        gradient_checkpointing: state.config.training.gradientCheckpointing,
        use_8bit_optimizer: state.config.training.use8BitOptimizer,
      },
      merge: true,
      merge_output_path: `${runDir}/merged`,
      export: {
        enabled: true,
        quant_type: 'Q4_K_M',
        gguf_output_path: ggufOutputPath,
      },
    };

    let engineHasError = false;

    try {
      const callbacks = createDefaultCallbacks({
        onStage: (msg) => {
          dispatch({ type: 'SET_TRAINING_STAGE', stage: msg.message });

          // Handle export stages
          if (msg.stage === 'merging') {
            dispatch({ type: 'SET_STEP', step: 'export' });
            dispatch({ type: 'SET_EXPORT_STAGE', id: 'merge', status: 'active' });
          }
        },
        onProgress: (msg) => {
          dispatch({ type: 'SET_TRAINING_PROGRESS', progress: msg });
          if (state.initialLoss === null && msg.loss > 0) {
            dispatch({ type: 'SET_INITIAL_LOSS', loss: msg.loss });
          }
        },
        onCheckpoint: (msg) => {
          dispatch({ type: 'SET_CHECKPOINT', path: msg.path });
        },
        onExportProgress: (msg) => {
          if (msg.stage === 'converting') {
            dispatch({ type: 'SET_EXPORT_STAGE', id: 'merge', status: 'complete' });
            dispatch({ type: 'SET_EXPORT_STAGE', id: 'convert', status: 'active' });
          } else if (msg.stage === 'quantizing') {
            dispatch({ type: 'SET_EXPORT_STAGE', id: 'convert', status: 'complete' });
            dispatch({ type: 'SET_EXPORT_STAGE', id: 'quantize', status: 'active' });
          } else if (msg.stage === 'complete') {
            dispatch({ type: 'SET_EXPORT_STAGE', id: 'quantize', status: 'complete' });
          }
        },
        onDone: async (msg) => {
          // Write Modelfile and import to Ollama
          dispatch({ type: 'SET_EXPORT_STAGE', id: 'import', status: 'active' });

          const finalGgufPath = msg.ggufPath ?? ggufOutputPath;
          const modelfilePath = await writeModelfile(runDir, finalGgufPath, state.model!);
          const ollamaResult = await importToOllama(modelName, modelfilePath);

          dispatch({
            type: 'SET_EXPORT_STAGE',
            id: 'import',
            status: ollamaResult.success ? 'complete' : 'error',
            errorMessage: ollamaResult.error,
          });
          dispatch({
            type: 'SET_RESULTS',
            adapterPath: msg.adapterPath ?? undefined,
            ggufPath: finalGgufPath,
            modelfilePath,
            ollamaModelName: modelName,
            ollamaImported: ollamaResult.success,
            elapsedSeconds: msg.elapsedSeconds,
          });
          dispatch({ type: 'SET_STEP', step: 'done' });
        },
        onError: (msg) => {
          engineHasError = true;
          dispatch({ type: 'SET_ERROR', message: msg.message, guidance: msg.detail });
        },
        onExit: (code) => {
          if (code !== 0 && !engineHasError && state.step !== 'done') {
            dispatch({
              type: 'SET_ERROR',
              message: 'The training engine exited unexpectedly.',
              guidance: state.lastCheckpoint
                ? `Your progress was saved at the last checkpoint. Run Orb again to resume.`
                : 'Try running Orb again. If this keeps happening, try a smaller model or batch size.',
            });
          }
        },
      });

      await spawnTrainingEngine(engineConfig, callbacks, { debug });
    } catch (err) {
      const e = toPlainLanguageError(err);
      dispatch({ type: 'SET_ERROR', message: e.userMessage, guidance: e.guidance });
    }
  }, [state.hardware, state.model, state.datasetMeta, state.config, state.hfToken, state.initialLoss, state.lastCheckpoint, state.step, debug]);

  // Determine the active tab
  const getActiveTab = (): 'model' | 'dataset' | 'train' | null => {
    switch (state.step) {
      case 'model': return 'model';
      case 'dataset': return 'dataset';
      case 'config':
      case 'dependencies':
      case 'training':
      case 'export':
      case 'done': return 'train';
      default: return null;
    }
  };

  const activeTab = getActiveTab();

  const renderStep = () => {
    switch (state.step) {
      case 'initializing':
        return (
          <Box flexDirection="column" padding={1}>
            <Spinner message="Detecting hardware..." />
          </Box>
        );

      case 'welcome':
        return (
          <Welcome
            hardware={state.hardware!}
            onContinue={() => dispatch({ type: 'SET_STEP', step: 'model' })}
          />
        );

      case 'model':
        return (
          <ModelStep
            hardware={state.hardware!}
            onComplete={(model, hfToken) => {
              dispatch({ type: 'SET_MODEL', model, hfToken });
              dispatch({ type: 'SET_STEP', step: 'dataset' });
            }}
            onBack={() => dispatch({ type: 'SET_STEP', step: 'model' })} // no back from model
          />
        );

      case 'dataset':
        return (
          <DatasetStep
            onComplete={(meta) => {
              dispatch({ type: 'SET_DATASET', meta });
              // Compute auto-config now that both model and dataset are known
              const config = computeAutoConfig(state.hardware!, state.model!, meta);
              dispatch({ type: 'SET_CONFIG', config });
              dispatch({ type: 'SET_STEP', step: 'config' });
            }}
            onBack={() => dispatch({ type: 'SET_STEP', step: 'model' })}
          />
        );

      case 'config':
        return (
          <ConfigStep
            config={state.config!}
            onConfirm={async (finalConfig) => {
              dispatch({ type: 'SET_CONFIG', config: finalConfig });

              // Check if dependencies need setup
              const ready = await areDependenciesReady();
              if (!ready) {
                dispatch({ type: 'SET_STEP', step: 'dependencies' });
                try {
                  await setupDependencies(state.hardware!.backend, {
                    onStatus: (msg) => setDepStatus(msg),
                    onProgress: (_step, _total, msg) => setDepStatus(msg),
                    onWarning: () => {},
                  });
                  // Dependencies ready, start training
                  await startTraining();
                } catch (err) {
                  const e = toPlainLanguageError(err);
                  dispatch({ type: 'SET_ERROR', message: e.userMessage, guidance: e.guidance });
                }
              } else {
                await startTraining();
              }
            }}
            onBack={() => dispatch({ type: 'SET_STEP', step: 'dataset' })}
          />
        );

      case 'dependencies':
        return (
          <Box flexDirection="column" padding={1}>
            <Text bold color={INK.primary}>FIRST-TIME SETUP</Text>
            <Text color={INK.dim}>This only happens once and may take a few minutes.</Text>
            <Box marginTop={1}>
              <Spinner message={depStatus || 'Installing dependencies...'} />
            </Box>
          </Box>
        );

      case 'training':
        return (
          <TrainingStep
            modelId={state.model!.id}
            totalSteps={state.config!.totalSteps}
            progress={state.trainingProgress}
            stage={state.trainingStage}
            lastCheckpoint={state.lastCheckpoint}
            initialLoss={state.initialLoss}
          />
        );

      case 'export':
        return (
          <ExportStep
            stages={state.exportStages}
            currentStage={state.exportStages.find((s) => s.status === 'active')?.label ?? ''}
          />
        );

      case 'done':
        return (
          <DoneStep
            ggufPath={state.ggufPath ?? ''}
            modelfilePath={state.modelfilePath ?? ''}
            modelName={state.ollamaModelName ?? ''}
            ollamaImported={state.ollamaImported}
            fileSize={state.ggufFileSize ?? ''}
            onExit={() => exit()}
          />
        );

      case 'error':
        return (
          <Box flexDirection="column" padding={1}>
            <ErrorBox
              message={state.error?.message ?? 'An unknown error occurred.'}
              guidance={state.error?.guidance}
            />
            {state.lastCheckpoint && (
              <Box marginTop={1}>
                <Text color={INK.gold}>
                  Your training progress was saved. Run Orb again to resume.
                </Text>
              </Box>
            )}
          </Box>
        );

      default:
        return <Text>Unknown state</Text>;
    }
  };

  const getMascotState = (): 'idle' | 'working' | 'done' => {
    switch (state.step) {
      case 'initializing':
      case 'dependencies':
      case 'training':
      case 'export':
        return 'working';
      case 'done':
        return 'done';
      default:
        // Could also be 'working' during dataset mapping/validation but we rely on sub-spinners there
        return 'idle';
    }
  };

  return (
    <Box flexDirection="column">
      {/* Orbital Terminal header — mascot + ORB wordmark */}
      <Box paddingX={1} paddingTop={1} paddingBottom={0} alignItems="center">
        <Mascot state={getMascotState()} />
        <Text bold color={INK.primary}>{' ORB'}</Text>
        <Text color={INK.faint}>{'  '}{glyph.separator.repeat(4)}{'  '}</Text>
        <Text color={INK.secondary}>{'AI MODEL FINE-TUNING'}</Text>
        <Text color={INK.faint}>{'  '}{glyph.separator.repeat(4)}{'  '}</Text>
        <Text color={INK.faint}>{'ORBITAL TERMINAL'}</Text>
      </Box>
      {activeTab && <TabHeader activeTab={activeTab} />}
      {renderStep()}
    </Box>
  );
}