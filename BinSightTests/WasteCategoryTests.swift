import Testing
@testable import BinSight

@MainActor
struct WasteCategoryTests {
    @Test("The six planned waste classes are declared, with no duplicates")
    func allCasesCoverSixDistinctClasses() {
        #expect(WasteCategory.allCases.count == 6)
        #expect(Set(WasteCategory.allCases.map(\.rawValue)).count == 6)
    }

    @Test("Every class has a distinct display name")
    func displayNamesAreDistinctAndPresent() {
        let names = WasteCategory.allCases.map(\.displayName)
        #expect(names.allSatisfy { !$0.isEmpty })
        #expect(Set(names).count == names.count)
    }

    @Test("Every class has its own SF Symbol, so two classes never look alike")
    func symbolsAreDistinctAndPresent() {
        let symbols = WasteCategory.allCases.map(\.symbolName)
        #expect(symbols.allSatisfy { !$0.isEmpty })
        #expect(Set(symbols).count == symbols.count)
    }

    @Test("Every class has generic disposal guidance")
    func guidanceIsPresent() {
        for category in WasteCategory.allCases {
            #expect(!category.disposalGuidance.isEmpty)
        }
    }
}
