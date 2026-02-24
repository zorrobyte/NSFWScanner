enum ScanState: Equatable {
    case idle
    case requestingPermission
    case scanning
    case reviewing
    case committingToAlbum
    case error(String)
}
