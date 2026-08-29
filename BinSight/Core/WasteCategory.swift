import Foundation

/// The six classes BinSight sorts a single item into.
///
/// Presentation metadata lives here so the symbol, label and guidance for a
/// class are stated once. The accent colour is the one piece that needs
/// SwiftUI, so it sits alongside the palette in `WasteCategory+Style`.
enum WasteCategory: String, CaseIterable, Identifiable, Sendable {
    case cardboard
    case glass
    case metal
    case paper
    case plastic
    case generalWaste

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .cardboard: L("waste.cardboard.name")
        case .glass: L("waste.glass.name")
        case .metal: L("waste.metal.name")
        case .paper: L("waste.paper.name")
        case .plastic: L("waste.plastic.name")
        case .generalWaste: L("waste.generalWaste.name")
        }
    }

    var symbolName: String {
        switch self {
        case .cardboard: "shippingbox.fill"
        case .glass: "wineglass.fill"
        case .metal: "cylinder.fill"
        case .paper: "newspaper.fill"
        case .plastic: "waterbottle.fill"
        case .generalWaste: "trash.fill"
        }
    }

    /// Generic handling advice — deliberately about the item itself, never
    /// about a particular council's or region's recycling rules. The UI labels
    /// it as generic so it is never mistaken for local law.
    var disposalGuidance: String {
        switch self {
        case .cardboard: L("waste.cardboard.guidance")
        case .glass: L("waste.glass.guidance")
        case .metal: L("waste.metal.guidance")
        case .paper: L("waste.paper.guidance")
        case .plastic: L("waste.plastic.guidance")
        case .generalWaste: L("waste.generalWaste.guidance")
        }
    }
}

extension WasteCategory {
    /// The order the model emits scores in — index 0 is output index 0.
    ///
    /// This must match `ml/artifacts/labels.txt` exactly. It is deliberately
    /// **not** alphabetical: sorted, `general_waste` would land at index 1
    /// instead of 5, and every prediction would be silently mislabelled.
    /// `WasteCategoryTests` asserts both the order and its non-alphabetical-ness.
    static let modelOutputOrder: [WasteCategory] = allCases

    /// The label string used in `labels.txt` and the model contract.
    ///
    /// The trained model emits TrashNet's own class name `trash` for the
    /// residual class; the app's internal name is `generalWaste`. Both spellings
    /// are accepted on the way in so a labels file from either source loads.
    var modelLabel: String {
        self == .generalWaste ? "trash" : rawValue
    }

    /// Resolves a label from `labels.txt` to a category.
    ///
    /// Returns `nil` for an unknown name rather than guessing — a labels file
    /// that does not match the model is a build error, and silently mapping it
    /// would mislabel every prediction.
    init?(modelLabel label: String) {
        let normalised = label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalised {
        case "cardboard": self = .cardboard
        case "glass": self = .glass
        case "metal": self = .metal
        case "paper": self = .paper
        case "plastic": self = .plastic
        // TrashNet's residual class. Shown as "General waste" and never
        // presented as a recyclable stream.
        case "trash", "general_waste", "generalwaste": self = .generalWaste
        default: return nil
        }
    }
}
