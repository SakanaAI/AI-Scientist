# Modular Architecture Redesign

This design document outlines the approach for implementing a modular architecture for the AI-Scientist project.

## 1. Overview

The current AI-Scientist architecture tightly couples LLM interactions, scientific reasoning, experiment execution, and output generation. This redesign aims to separate these concerns into distinct modules with well-defined interfaces, enabling better maintainability, extensibility, and testing.

## 2. Design Goals

- **Separation of Concerns**: Isolate different aspects of the system into specialized components
- **Interface Stability**: Define clear interfaces between components
- **Extensibility**: Make it easy to add new modules and capabilities
- **Testability**: Enable comprehensive testing of individual components
- **Configuration**: Support hierarchical, validated configuration

## 3. Core Architecture Components

### 3.1 LLM Service Layer

**Purpose**: Manage all interactions with language models, abstracting their implementation details.

**Components**:
- `LLMClientFactory`: Creates appropriate LLM clients based on configuration
- `LLMClient`: Abstract interface for LLM interactions
- `PromptManager`: Handles prompt templates and interpolation
- `ResponseParser`: Extracts structured data from LLM responses

**Interfaces**:
```python
class LLMClient:
    async def complete(self, prompt: str, options: Dict[str, Any]) -> str:
        """Generate a completion from the LLM."""
        pass
    
    async def chat(self, messages: List[Dict[str, str]], options: Dict[str, Any]) -> str:
        """Generate a chat response from the LLM."""
        pass

class PromptManager:
    def load_template(self, template_name: str) -> Template:
        """Load a prompt template by name."""
        pass
    
    def render(self, template: Template, variables: Dict[str, Any]) -> str:
        """Render a template with variables."""
        pass
```

### 3.2 Scientific Core

**Purpose**: Implement the scientific reasoning processes, independent of LLM implementation.

**Components**:
- `HypothesisGenerator`: Creates scientific hypotheses
- `ExperimentDesigner`: Designs experiments to test hypotheses
- `ResultAnalyzer`: Analyzes experimental results
- `LiteratureReviewer`: Conducts literature reviews

**Interfaces**:
```python
class HypothesisGenerator:
    async def generate(self, domain: str, constraints: Dict[str, Any]) -> List[Hypothesis]:
        """Generate scientific hypotheses based on domain and constraints."""
        pass

class ExperimentDesigner:
    async def design(self, hypothesis: Hypothesis, resources: Resources) -> Experiment:
        """Design an experiment to test a hypothesis given available resources."""
        pass

class ResultAnalyzer:
    async def analyze(self, experiment: Experiment, results: ExperimentResults) -> Analysis:
        """Analyze the results of an experiment."""
        pass
```

### 3.3 Experiment Execution

**Purpose**: Handle the execution and monitoring of experiments.

**Components**:
- `ExperimentRunner`: Executes experiments
- `ResourceManager`: Manages computation resources
- `ExecutionMonitor`: Monitors and logs experiment execution
- `ResultCollector`: Collects and structures results

**Interfaces**:
```python
class ExperimentRunner:
    async def run(self, experiment: Experiment, resources: Resources) -> ExperimentResults:
        """Run an experiment with the given resources."""
        pass
    
    def stop(self, experiment_id: str) -> None:
        """Stop a running experiment."""
        pass

class ResourceManager:
    def allocate(self, requirements: ResourceRequirements) -> Resources:
        """Allocate resources based on requirements."""
        pass
    
    def release(self, resources: Resources) -> None:
        """Release allocated resources."""
        pass
```

### 3.4 Paper Generation

**Purpose**: Generate scientific papers from experimental results.

**Components**:
- `PaperStructurer`: Defines the structure of a paper
- `SectionGenerator`: Generates content for paper sections
- `CitationManager`: Manages citations and references
- `LaTeXGenerator`: Generates LaTeX output

**Interfaces**:
```python
class PaperStructurer:
    def structure(self, experiment: Experiment, analysis: Analysis) -> PaperStructure:
        """Create a structure for a paper based on experiment and analysis."""
        pass

class SectionGenerator:
    async def generate(self, section_type: str, data: Dict[str, Any]) -> str:
        """Generate content for a specific section type."""
        pass
```

### 3.5 Configuration Management

**Purpose**: Manage system-wide and component-specific configuration.

**Components**:
- `ConfigLoader`: Loads configuration from files
- `ConfigValidator`: Validates configuration against schemas
- `ConfigRegistry`: Maintains a registry of configuration

**Interfaces**:
```python
class ConfigLoader:
    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from a file."""
        pass
    
    def merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations with override precedence."""
        pass

class ConfigValidator:
    def validate(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate configuration against a schema."""
        pass
```

## 4. Communication Pattern

Components will communicate using a message-passing pattern:

1. **Synchronous Direct Calls**: For simple, immediate operations
2. **Asynchronous Events**: For operations that may take time or trigger side effects
3. **Queue-Based Processing**: For operations that need to be processed in order or retried

### Example: Event-Based Communication

```python
# Event definitions
class HypothesisGeneratedEvent:
    def __init__(self, hypothesis_id: str, hypothesis: Hypothesis):
        self.hypothesis_id = hypothesis_id
        self.hypothesis = hypothesis

# EventBus for communication
class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
    
    def subscribe(self, event_type: Type, handler: Callable):
        self.subscribers[event_type].append(handler)
    
    def publish(self, event: Any):
        for handler in self.subscribers[type(event)]:
            handler(event)
```

## 5. Plugin Architecture

The system will support plugins for domain-specific templates, algorithms, and integrations.

### Plugin Structure

```
plugin/
├── manifest.json      # Plugin metadata and requirements
├── models/            # Domain-specific models
├── templates/         # Prompt templates
├── experiments/       # Experiment definitions
├── algorithms/        # Custom algorithms
└── __init__.py        # Plugin entry point
```

### Plugin Registry

```python
class PluginRegistry:
    def __init__(self):
        self.plugins = {}
    
    def register(self, plugin_id: str, plugin: Plugin):
        self.plugins[plugin_id] = plugin
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        return self.plugins.get(plugin_id)
    
    def list_plugins(self) -> List[str]:
        return list(self.plugins.keys())
```

## 6. Implementation Strategy

### 6.1 Phase 1: Core Interfaces

1. Define interfaces for all components
2. Create a minimal implementation of the LLM Service Layer
3. Establish the event-based communication system
4. Implement configuration management

### 6.2 Phase 2: Component Migration

1. Refactor existing code into the new architecture, one component at a time
2. Start with the scientific core components
3. Implement basic unit tests for each component
4. Create simple integration tests

### 6.3 Phase 3: Plugin System

1. Implement the plugin registry
2. Create plugin discovery mechanism
3. Convert one existing template to a plugin as a proof of concept
4. Create documentation for plugin development

## 7. Directory Structure

```
ai_scientist/
├── core/                      # Core interfaces and base classes
│   ├── llm/                   # LLM service layer
│   ├── science/               # Scientific core
│   ├── execution/             # Experiment execution
│   ├── paper/                 # Paper generation
│   └── config/                # Configuration management
├── plugins/                   # Plugin system
│   ├── registry.py            # Plugin registry
│   ├── loader.py              # Plugin discovery and loading
│   └── base.py                # Base classes for plugins
├── implementations/           # Concrete implementations
│   ├── llm/                   # Specific LLM implementations
│   ├── science/               # Scientific algorithms
│   ├── execution/             # Execution engines
│   └── paper/                 # Paper generators
├── templates/                 # Core templates
│   ├── prompts/               # Prompt templates
│   └── papers/                # Paper templates
├── utils/                     # Utility functions
├── data/                      # Data models and schemas
└── api/                       # Public API
```

## 8. Testing Strategy

### 8.1 Unit Tests

- Test each component in isolation
- Mock dependencies using well-defined interfaces
- Aim for high test coverage (>80%)

### 8.2 Integration Tests

- Test interactions between components
- Use in-memory implementations for external services
- Test complete workflows with simplified scenarios

### 8.3 End-to-End Tests

- Test complete research cycles
- Use simplified research questions with known outcomes
- Monitor for regressions in capabilities

## 9. Performance Considerations

- Implement caching for LLM responses to reduce costs
- Use asynchronous programming for better concurrency
- Implement resource pooling and scheduling
- Consider scaling strategies for computation-intensive tasks

## 10. Migration Plan

1. Create the new architecture in parallel with existing code
2. Begin with basic implementations of each component
3. Gradually route functionality through the new components
4. Convert existing templates to plugins once the system is stable
5. Phase out legacy code as new capabilities match or exceed it

## 11. Open Questions and Risks

- How to balance flexibility with performance?
- How to handle backward compatibility with existing experiments?
- What is the right granularity for components?
- How to effectively test LLM-based reasoning?

## 12. Next Steps

1. Define detailed interface specifications for core components
2. Implement a prototype of the LLM service layer
3. Create configuration management system
4. Begin refactoring existing code to use new interfaces 