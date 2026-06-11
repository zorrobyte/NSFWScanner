import SwiftUI

@main
struct NSFWScannerApp: App {
    @State private var orchestrator = ScanOrchestrator()
    @State private var sortOrchestrator = SortOrchestrator()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(orchestrator)
                .environment(sortOrchestrator)
        }
        .defaultSize(width: 1000, height: 700)
    }
}
