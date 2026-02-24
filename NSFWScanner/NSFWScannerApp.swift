import SwiftUI

@main
struct NSFWScannerApp: App {
    @State private var orchestrator = ScanOrchestrator()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(orchestrator)
        }
        .defaultSize(width: 1000, height: 700)
    }
}
