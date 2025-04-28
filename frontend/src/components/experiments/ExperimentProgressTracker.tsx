import React from 'react';
import { 
  Box, 
  Stepper, 
  Step, 
  StepLabel, 
  Typography, 
  Paper, 
  LinearProgress,
  styled,
  StepIconProps,
  StepConnector,
  stepConnectorClasses
} from '@mui/material';
import { Check, ScienceOutlined, ErrorOutline } from '@mui/icons-material';

interface ExperimentStepProps {
  label: string;
  description: string;
  status: 'completed' | 'in-progress' | 'pending' | 'failed';
  estimatedTimeRemaining?: string;
  completedAt?: string;
}

interface ExperimentProgressTrackerProps {
  steps: ExperimentStepProps[];
  currentStep: number;
  overallProgress: number;
}

const QontoConnector = styled(StepConnector)(({ theme }) => ({
  [`&.${stepConnectorClasses.alternativeLabel}`]: {
    top: 10,
    left: 'calc(-50% + 16px)',
    right: 'calc(50% + 16px)',
  },
  [`&.${stepConnectorClasses.active}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      borderColor: theme.palette.primary.main,
    },
  },
  [`&.${stepConnectorClasses.completed}`]: {
    [`& .${stepConnectorClasses.line}`]: {
      borderColor: theme.palette.primary.main,
    },
  },
  [`& .${stepConnectorClasses.line}`]: {
    borderColor: theme.palette.mode === 'dark' ? theme.palette.grey[800] : '#eaeaf0',
    borderTopWidth: 3,
    borderRadius: 1,
  },
}));

const QontoStepIconRoot = styled('div')<{ ownerState: { active?: boolean } }>(
  ({ theme, ownerState }) => ({
    color: theme.palette.mode === 'dark' ? theme.palette.grey[700] : '#eaeaf0',
    display: 'flex',
    height: 22,
    alignItems: 'center',
    ...(ownerState.active && {
      color: theme.palette.primary.main,
    }),
    '& .QontoStepIcon-completedIcon': {
      color: theme.palette.primary.main,
      zIndex: 1,
      fontSize: 18,
    },
    '& .QontoStepIcon-circle': {
      width: 8,
      height: 8,
      borderRadius: '50%',
      backgroundColor: 'currentColor',
    },
    '& .QontoStepIcon-inProgress': {
      color: theme.palette.primary.main,
      zIndex: 1,
      fontSize: 18,
    },
    '& .QontoStepIcon-error': {
      color: theme.palette.error.main,
      zIndex: 1,
      fontSize: 18,
    },
  }),
);

function QontoStepIcon(props: StepIconProps & { status?: string }) {
  const { active, completed, className, status } = props;

  return (
    <QontoStepIconRoot ownerState={{ active }} className={className}>
      {completed ? (
        <Check className="QontoStepIcon-completedIcon" />
      ) : status === 'failed' ? (
        <ErrorOutline className="QontoStepIcon-error" />
      ) : active ? (
        <ScienceOutlined className="QontoStepIcon-inProgress" />
      ) : (
        <div className="QontoStepIcon-circle" />
      )}
    </QontoStepIconRoot>
  );
}

const ExperimentProgressTracker: React.FC<ExperimentProgressTrackerProps> = ({ 
  steps, 
  currentStep,
  overallProgress 
}) => {
  return (
    <Paper sx={{ p: 3, mb: 3 }} elevation={0} variant="outlined">
      <Box sx={{ mb: 3 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="h6">Experiment Progress</Typography>
          <Typography variant="body2" color="text.secondary">
            {Math.round(overallProgress)}% Complete
          </Typography>
        </Box>
        <LinearProgress 
          variant="determinate" 
          value={overallProgress} 
          sx={{ height: 8, borderRadius: 4 }}
        />
      </Box>
      
      <Stepper 
        activeStep={currentStep} 
        alternativeLabel 
        connector={<QontoConnector />}
      >
        {steps.map((step, index) => (
          <Step key={step.label} completed={step.status === 'completed'}>
            <StepLabel 
              StepIconComponent={(iconProps) => <QontoStepIcon {...iconProps} status={step.status} />}
            >
              <Typography 
                variant="subtitle2" 
                color={
                  step.status === 'failed' 
                    ? 'error' 
                    : index === currentStep && step.status === 'in-progress'
                      ? 'primary'
                      : 'text.primary'
                }
              >
                {step.label}
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block">
                {step.status === 'completed' && step.completedAt 
                  ? `Completed at ${step.completedAt}`
                  : step.status === 'in-progress' && step.estimatedTimeRemaining
                    ? `Est. remaining: ${step.estimatedTimeRemaining}`
                    : step.description
                }
              </Typography>
            </StepLabel>
          </Step>
        ))}
      </Stepper>
      
      <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Typography variant="body2" color="text.secondary">
          {steps[currentStep]?.status === 'in-progress' 
            ? `Currently working on: ${steps[currentStep]?.label} - ${steps[currentStep]?.description}`
            : steps[currentStep]?.status === 'failed'
              ? `Failed at: ${steps[currentStep]?.label} - ${steps[currentStep]?.description}`
              : steps[currentStep]?.status === 'completed' && steps.length === currentStep + 1
                ? `All steps completed successfully`
                : `Next step: ${steps[currentStep]?.label} - ${steps[currentStep]?.description}`
          }
        </Typography>
      </Box>
    </Paper>
  );
};

export default ExperimentProgressTracker; 