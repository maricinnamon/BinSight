import SwiftUI

extension WasteCategory {
    /// The one accent colour for this class, used for its icon, its label and
    /// its confidence bar. Defined here rather than in any view so a class
    /// never drifts to a different colour on a different screen.
    ///
    /// Every value is picked to stay legible on `BinSightTheme.surface`, which
    /// is dark in both light and dark appearance.
    var accent: Color {
        switch self {
        case .cardboard: BinSightTheme.amber
        case .glass: BinSightTheme.mint
        case .metal: BinSightTheme.violet
        case .paper: BinSightTheme.lime
        case .plastic: BinSightTheme.coral
        case .generalWaste: BinSightTheme.slate
        }
    }
}
