/// Runs `work` in a task that is cancelled before the work starts (#80), and answers whether the work threw `CancellationError`. The
/// kit's long loops check the calling task (`Task.checkCancellation`) per row block or per pass, so a cancelled call stops there.
package func throwsCancellation<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async -> Bool {
    let task = Task { () throws -> T in
        withUnsafeCurrentTask { $0?.cancel() }
        return try work()
    }
    do {
        _ = try await task.value
        return false
    } catch is CancellationError {
        return true
    } catch {
        return false
    }
}
