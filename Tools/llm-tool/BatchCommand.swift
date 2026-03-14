// Copyright © 2025 Apple Inc.

import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

struct BatchCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "batch",
        abstract: "process multiple prompts concurrently using batched generation"
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var generate: GenerateArguments

    @Option(
        name: .shortAndLong,
        help: "Prompts to process (repeatable, at least 1 required)"
    )
    var prompt: [String]

    @Option(
        name: .long,
        help: "Maximum number of concurrent prompts in a batch (default: all)"
    )
    var batchSize: Int?

    mutating func validate() throws {
        if prompt.isEmpty {
            throw ValidationError("At least one --prompt is required.")
        }
    }

    @MainActor
    mutating func run() async throws {
        let defaultModel = MLXLLM.LLMRegistry.mistral7B4bit

        // Load model
        let modelContainer = try await memory.start { [args] in
            try await args.load(
                defaultModel: defaultModel.name, modelFactory: LLMModelFactory.shared)
        }

        // Attach an InferenceScheduler for batching support
        let scheduler = InferenceScheduler()
        modelContainer.scheduler = scheduler

        // Update context/configuration with command line parameters
        await modelContainer.update { [generate] context in
            generate.prepare(&context)
        }

        let modelConfiguration = await modelContainer.configuration

        if !generate.quiet {
            print("Loaded \(modelConfiguration.name)")
            print("Processing \(prompt.count) prompt(s)...\n")
        }

        let allPrompts = prompt
        let maxConcurrent = batchSize ?? allPrompts.count
        let generateParams = generate.generateParameters

        let overallStart = Date()

        // Collect results: (promptIndex, output, info)
        let results = ResultCollector()

        // Process prompts in batches of maxConcurrent
        var promptIndex = 0
        while promptIndex < allPrompts.count {
            let batchEnd = min(promptIndex + maxConcurrent, allPrompts.count)
            let batchPrompts = Array(allPrompts[promptIndex..<batchEnd])
            let batchStartIndex = promptIndex

            await withTaskGroup(of: Void.self) { group in
                for (offset, promptText) in batchPrompts.enumerated() {
                    let idx = batchStartIndex + offset
                    group.addTask { @Sendable in
                        do {
                            // Prepare input
                            let messages: [[String: String]] = [
                                ["role": "user", "content": promptText]
                            ]
                            let tokens = try await modelContainer.applyChatTemplate(
                                messages: messages)
                            let lmInput = LMInput(tokens: MLXArray(tokens))

                            // Generate via the scheduler-backed ModelContainer
                            let stream = try await modelContainer.generate(
                                input: lmInput,
                                parameters: generateParams
                            )

                            var output = ""
                            var completionInfo: GenerateCompletionInfo?
                            for await generation in stream {
                                switch generation {
                                case .chunk(let text):
                                    output += text
                                case .info(let info):
                                    completionInfo = info
                                case .toolCall:
                                    break
                                }
                            }

                            await results.add(
                                index: idx,
                                output: output,
                                info: completionInfo
                            )
                        } catch {
                            await results.add(
                                index: idx,
                                output: "[Error: \(error.localizedDescription)]",
                                info: nil
                            )
                        }
                    }
                }
            }

            promptIndex = batchEnd
        }

        let overallElapsed = Date().timeIntervalSince(overallStart)

        // Print results
        let collected = await results.all()
        let sorted = collected.sorted { $0.index < $1.index }

        for entry in sorted {
            print("--- Prompt \(entry.index + 1) ---")
            print("Input:  \(allPrompts[entry.index])")
            print("Output: \(entry.output)")
            if let info = entry.info {
                print("Stats:  \(info.summary())")
            }
            print()
        }

        if !generate.quiet {
            // Aggregate statistics
            var totalPromptTokens = 0
            var totalGenTokens = 0
            for entry in sorted {
                if let info = entry.info {
                    totalPromptTokens += info.promptTokenCount
                    totalGenTokens += info.generationTokenCount
                }
            }

            print("=== Batch Summary ===")
            print("Prompts:           \(allPrompts.count)")
            print("Total prompt tokens:  \(totalPromptTokens)")
            print("Total gen tokens:     \(totalGenTokens)")
            print(
                "Wall-clock time:      \(String(format: "%.3f", overallElapsed))s"
            )
            if overallElapsed > 0 {
                print(
                    "Aggregate tok/s:      \(String(format: "%.1f", Double(totalGenTokens) / overallElapsed))"
                )
            }

            memory.reportMemoryStatistics()
        }
    }
}

// Thread-safe result collector
private actor ResultCollector {
    struct Entry {
        let index: Int
        let output: String
        let info: GenerateCompletionInfo?
    }

    private var entries: [Entry] = []

    func add(index: Int, output: String, info: GenerateCompletionInfo?) {
        entries.append(Entry(index: index, output: output, info: info))
    }

    func all() -> [Entry] {
        entries
    }
}
