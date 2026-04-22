"""
VBVR-Bench Evaluators Module

This module provides task-specific evaluators for all 100 VBVR-Bench tasks.
Each evaluator implements rule-based evaluation following documented criteria.
"""

from typing import Any, Dict, Optional

from .base_evaluator import BaseEvaluator

# In-Domain_50 Part 1 (10 classes)
from .In_Domain_50_part1 import (
    GridAvoidObstaclesEvaluator as InDomainGridAvoidObstaclesEvaluator,
)
from .In_Domain_50_part1 import (
    GridGoThroughBlockEvaluator as InDomainGridGoThroughBlockEvaluator,
)
from .In_Domain_50_part1 import (
    GridNumberSequenceEvaluator as InDomainGridNumberSequenceEvaluator,
)
from .In_Domain_50_part1 import (
    GridShortestPathEvaluator as InDomainGridShortestPathEvaluator,
)
from .In_Domain_50_part1 import (
    IdentifyObjectsInRegionEvaluator as InDomainIdentifyObjectsInRegionEvaluator,
)
from .In_Domain_50_part1 import (
    MultiObjectPlacementEvaluator as InDomainMultiObjectPlacementEvaluator,
)
from .In_Domain_50_part1 import (
    MultipleOcclusionsVerticalEvaluator as InDomainMultipleOcclusionsEvaluator,
)
from .In_Domain_50_part1 import (
    SeparateObjectsSpinningEvaluator as InDomainSeparateObjectsSpinningEvaluator,
)
from .In_Domain_50_part1 import StableSortEvaluator as InDomainStableSortEvaluator
from .In_Domain_50_part1 import (
    TrackObjectMovementEvaluator as InDomainTrackObjectMovementEvaluator,
)

# In-Domain_50 Part 2 (10 classes)
from .In_Domain_50_part2 import (
    AttentionShiftEvaluator as InDomainAttentionShiftEvaluator,
)
from .In_Domain_50_part2 import ChartExtremeEvaluator as InDomainChartExtremeEvaluator
from .In_Domain_50_part2 import (
    DirectedGraphNavigationEvaluator as InDomainDirectedGraphNavigationEvaluator,
)
from .In_Domain_50_part2 import (
    GridHighestCostEvaluator as InDomainGridHighestCostEvaluator,
)
from .In_Domain_50_part2 import (
    KeyDoorMatchingEvaluator as InDomainKeyDoorMatchingEvaluator,
)
from .In_Domain_50_part2 import (
    PredictNextColorEvaluator as InDomainPredictNextColorEvaluator,
)
from .In_Domain_50_part2 import (
    SelectNextFigureIncreasingEvaluator as InDomainSelectNextFigureIncreasingEvaluator,
)
from .In_Domain_50_part2 import (
    SelectNextFigureLargeSmallEvaluator as InDomainSelectNextFigureLargeSmallEvaluator,
)
from .In_Domain_50_part2 import (
    SpotUniqueColorEvaluator as InDomainSpotUniqueColorEvaluator,
)
from .In_Domain_50_part2 import (
    UnderstandSceneStructureEvaluator as InDomainUnderstandSceneStructureEvaluator,
)

# In-Domain_50 Part 3 (10 classes)
from .In_Domain_50_part3 import BallBounceEvaluator as InDomainBallBounceEvaluator
from .In_Domain_50_part3 import ColorAdditionEvaluator as InDomainColorAdditionEvaluator
from .In_Domain_50_part3 import (
    ConstructConcentricRingEvaluator as InDomainConstructConcentricRingEvaluator,
)
from .In_Domain_50_part3 import (
    GlassRefractionEvaluator as InDomainGlassRefractionEvaluator,
)
from .In_Domain_50_part3 import (
    IdentifyAllHollowPointsEvaluator as InDomainIdentifyAllHollowPointsEvaluator,
)
from .In_Domain_50_part3 import (
    MirrorReflectionEvaluator as InDomainMirrorReflectionEvaluator,
)
from .In_Domain_50_part3 import (
    ShapeColorThenScaleEvaluator as InDomainShapeColorThenScaleEvaluator,
)
from .In_Domain_50_part3 import (
    ShapeOutlineFillEvaluator as InDomainShapeOutlineFillEvaluator,
)
from .In_Domain_50_part3 import (
    ShapeOutlineThenMoveEvaluator as InDomainShapeOutlineThenMoveEvaluator,
)
from .In_Domain_50_part3 import (
    ShapeScaleThenOutlineEvaluator as InDomainShapeScaleThenOutlineEvaluator,
)

# In-Domain_50 Part 4 (10 classes)
from .In_Domain_50_part4 import BallColorEvaluator as InDomainBallColorEvaluator
from .In_Domain_50_part4 import BallEatingEvaluator as InDomainBallEatingEvaluator
from .In_Domain_50_part4 import BookshelfEvaluator as InDomainBookshelfEvaluator
from .In_Domain_50_part4 import (
    ConstructionBlueprintEvaluator as InDomainConstructionBlueprintEvaluator,
)
from .In_Domain_50_part4 import (
    CountingObjectEvaluator as InDomainCountingObjectEvaluator,
)
from .In_Domain_50_part4 import (
    DominoChainBranchEvaluator as InDomainDominoChainBranchEvaluator,
)
from .In_Domain_50_part4 import (
    DominoChainGapEvaluator as InDomainDominoChainGapEvaluator,
)
from .In_Domain_50_part4 import DotToDotEvaluator as InDomainDotToDotEvaluator
from .In_Domain_50_part4 import (
    LEGOConstructionEvaluator as InDomainLEGOConstructionEvaluator,
)
from .In_Domain_50_part4 import RollingBallEvaluator as InDomainRollingBallEvaluator

# In-Domain_50 Part 5 (10 classes)
from .In_Domain_50_part5 import ClockTimeEvaluator as InDomainClockTimeEvaluator
from .In_Domain_50_part5 import (
    CommunicatingVesselsEvaluator as InDomainCommunicatingVesselsEvaluator,
)
from .In_Domain_50_part5 import GridShiftEvaluator as InDomainGridShiftEvaluator
from .In_Domain_50_part5 import LightSequenceEvaluator as InDomainLightSequenceEvaluator
from .In_Domain_50_part5 import MajorityColorEvaluator as InDomainMajorityColorEvaluator
from .In_Domain_50_part5 import RotationEvaluator as InDomainRotationEvaluator
from .In_Domain_50_part5 import (
    RotationPuzzleEvaluator as InDomainRotationPuzzleEvaluator,
)
from .In_Domain_50_part5 import (
    SequenceCompletionEvaluator as InDomainSequenceCompletionEvaluator,
)
from .In_Domain_50_part5 import SlidingPuzzleEvaluator as InDomainSlidingPuzzleEvaluator
from .In_Domain_50_part5 import TrafficLightEvaluator as InDomainTrafficLightEvaluator

# Out-of-Domain_50 Part 1 (10 classes)
from .Out_of_Domain_50_part1 import (
    CircleLargestNumericalValueEvaluator as OutOfDomainCircleLargestEvaluator,
)
from .Out_of_Domain_50_part1 import (
    ConnectingColorEvaluator as InDomainConnectingColorEvaluator,
)
from .Out_of_Domain_50_part1 import (
    IdentifyUniqueFigureEvaluator as OutOfDomainIdentifyUniqueFigureEvaluator,
)
from .Out_of_Domain_50_part1 import (
    LocatePointInOverlappingAreaEvaluator,
)
from .Out_of_Domain_50_part1 import (
    LocateTopmostFigureEvaluator as OutOfDomainLocateTopmostFigureEvaluator,
)
from .Out_of_Domain_50_part1 import (
    MarkSecondLargestShapeEvaluator as OutOfDomainMarkSecondLargestEvaluator,
)
from .Out_of_Domain_50_part1 import (
    MultipleKeysForOneDoorEvaluator,
)
from .Out_of_Domain_50_part1 import (
    SelectLongestPolygonSideEvaluator as OutOfDomainSelectLongestSideEvaluator,
)
from .Out_of_Domain_50_part1 import (
    SelectNextFigureAlternatingEvaluator,
)
from .Out_of_Domain_50_part1 import (
    SeparateObjectsNoSpinEvaluator as InDomainSeparateObjectsNoSpinEvaluator,
)

# Out-of-Domain_50 Part 2 (10 classes)
from .Out_of_Domain_50_part2 import (
    ArrangeCirclesByCircumferenceEvaluator as OutOfDomainArrangeCirclesEvaluator,
)
from .Out_of_Domain_50_part2 import (
    CircleCentralDotEvaluator,
)
from .Out_of_Domain_50_part2 import (
    DrawMidpointPerpendicularEvaluator as InDomainDrawMidpointPerpendicularEvaluator,
)
from .Out_of_Domain_50_part2 import (
    DrawNextSizedShapeEvaluator as OutOfDomainDrawNextShapeEvaluator,
)
from .Out_of_Domain_50_part2 import (
    FindIncorrectArrowDirectionEvaluator,
    IdentifyLargestAngleEvaluator,
)
from .Out_of_Domain_50_part2 import (
    IdentifyNearestSquareRectangleEvaluator as InDomainIdentifyNearestSquareRectangleEvaluator,
)
from .Out_of_Domain_50_part2 import (
    IdentifyPentagonsEvaluator as InDomainIdentifyPentagonsEvaluator,
)
from .Out_of_Domain_50_part2 import (
    LocateSegmentIntersectionEvaluator as InDomainLocateSegmentIntersectionEvaluator,
)
from .Out_of_Domain_50_part2 import (
    MarkWavePeaksEvaluator,
)

# Out-of-Domain_50 Part 3 (10 classes)
from .Out_of_Domain_50_part3 import (
    AddBordersToUnborderedEvaluator,
)
from .Out_of_Domain_50_part3 import (
    ColorTripleIntersectionEvaluator as InDomainColorTripleIntersectionEvaluator,
)
from .Out_of_Domain_50_part3 import (
    HighDensityLiquidEvaluator,
)
from .Out_of_Domain_50_part3 import (
    HighlightHorizontalLinesEvaluator as InDomainHighlightHorizontalLinesEvaluator,
)
from .Out_of_Domain_50_part3 import (
    IdentifyChineseCharacterEvaluator,
    MarkAsymmetricalShapeEvaluator,
)
from .Out_of_Domain_50_part3 import (
    MarkTangentPointEvaluator as InDomainMarkTangentPointEvaluator,
)
from .Out_of_Domain_50_part3 import (
    OutlineInnermostSquareEvaluator,
)
from .Out_of_Domain_50_part3 import (
    PigmentColorMixingEvaluator as OutOfDomainPigmentMixingEvaluator,
)
from .Out_of_Domain_50_part3 import (
    SelectLeftmostShapeEvaluator,
)

# Out-of-Domain_50 Part 4 (10 classes)
from .Out_of_Domain_50_part4 import (
    ConstructionStackEvaluator as OutOfDomainConstructionStackEvaluator,
)
from .Out_of_Domain_50_part4 import (
    GeometricTransformationEvaluator as OutOfDomainGeometricTransformEvaluator,
)
from .Out_of_Domain_50_part4 import MazePathfindingEvaluator as OutOfDomainMazeEvaluator
from .Out_of_Domain_50_part4 import (
    MoveObjectsToTargetEvaluator as InDomainMoveObjectsToTargetEvaluator,
)
from .Out_of_Domain_50_part4 import (
    ObjectSubtractionEvaluator as OutOfDomainObjectSubtractionEvaluator,
)
from .Out_of_Domain_50_part4 import (
    ShapeColorThenMoveEvaluator as OutOfDomainShapeColorMoveEvaluator,
)
from .Out_of_Domain_50_part4 import (
    ShapeScalingAnalogyEvaluator as OutOfDomainShapeScalingEvaluator,
)
from .Out_of_Domain_50_part4 import (
    ShapeSorterEvaluator as OutOfDomainShapeSorterEvaluator,
)
from .Out_of_Domain_50_part4 import (
    SymbolDeletionEvaluator as OutOfDomainSymbolDeletionEvaluator,
)
from .Out_of_Domain_50_part4 import (
    SymmetryCompletionEvaluator as OutOfDomainSymmetryEvaluator,
)

# Out-of-Domain_50 Part 5 (10 classes)
from .Out_of_Domain_50_part5 import (
    AnimalMatchingEvaluator as OutOfDomainAnimalMatchingEvaluator,
)
from .Out_of_Domain_50_part5 import (
    AnimalSizeSortingEvaluator as OutOfDomainAnimalSizeSortingEvaluator,
)
from .Out_of_Domain_50_part5 import (
    ControlPanelEvaluator as OutOfDomainControlPanelEvaluator,
)
from .Out_of_Domain_50_part5 import (
    GravityPhysicsEvaluator as OutOfDomainGravityPhysicsEvaluator,
)
from .Out_of_Domain_50_part5 import (
    ObjectRotation2DEvaluator as OutOfDomainObjectRotation2DEvaluator,
)
from .Out_of_Domain_50_part5 import (
    RavenMatrixEvaluator as OutOfDomainRavenMatrixEvaluator,
)
from .Out_of_Domain_50_part5 import (
    SymbolDeleteEvaluator as OutOfDomainSymbolDeleteEvaluator,
)
from .Out_of_Domain_50_part5 import (
    SymbolEditConstraintEvaluator as OutOfDomainSymbolEditEvaluator,
)
from .Out_of_Domain_50_part5 import (
    SymbolInsertEvaluator as OutOfDomainSymbolInsertEvaluator,
)
from .Out_of_Domain_50_part5 import (
    SymbolSubstituteEvaluator as OutOfDomainSymbolSubstituteEvaluator,
)

# ============================================================
# Import In-Domain_50 evaluators
# ============================================================






# ============================================================
# Import Out-of-Domain_50 evaluators
# ============================================================







# Task to evaluator mapping
TASK_EVALUATOR_MAP = {
    # In-Domain_50 Tasks (50 tasks)
    "G-3_stable_sort_data-generator": InDomainStableSortEvaluator,
    "G-13_grid_number_sequence_data-generator": InDomainGridNumberSequenceEvaluator,
    "G-15_grid_avoid_obstacles_data-generator": InDomainGridAvoidObstaclesEvaluator,
    "G-16_grid_go_through_block_data-generator": InDomainGridGoThroughBlockEvaluator,
    "G-18_grid_shortest_path_data-generator": InDomainGridShortestPathEvaluator,
    "G-21_multiple_occlusions_vertical_data-generator": InDomainMultipleOcclusionsEvaluator,
    "G-25_seperate_object_spinning_data-generator": InDomainSeparateObjectsSpinningEvaluator,
    "G-29_chart_extreme_with_data_data-generator": InDomainChartExtremeEvaluator,
    "G-31_directed_graph_navigation_data-generator": InDomainDirectedGraphNavigationEvaluator,
    "G-39_attention_shift_different_data-generator": InDomainAttentionShiftEvaluator,
    "G-41_grid_highest_cost_data-generator": InDomainGridHighestCostEvaluator,
    "G-43_understand_scene_structure_data-generator": InDomainUnderstandSceneStructureEvaluator,
    "G-45_key_door_matching_data-generator": InDomainKeyDoorMatchingEvaluator,
    "G-51_predict_next_color_data-generator": InDomainPredictNextColorEvaluator,
    "G-131_select_next_figure_increasing_size_sequence_data-generator": InDomainSelectNextFigureIncreasingEvaluator,
    "G-134_select_next_figure_large_small_alternating_sequence_data-generator": InDomainSelectNextFigureLargeSmallEvaluator,
    "G-138_spot_unique_non_repeated_color_data-generator": InDomainSpotUniqueColorEvaluator,
    "G-158_identify_all_hollow_points_data-generator": InDomainIdentifyAllHollowPointsEvaluator,
    "G-194_construct_concentric_ring_data-generator": InDomainConstructConcentricRingEvaluator,
    "G-5_multi_object_placement_data-generator": InDomainMultiObjectPlacementEvaluator,
    "G-8_track_object_movement_data-generator": InDomainTrackObjectMovementEvaluator,
    "G-9_identify_objects_in_region_data-generator": InDomainIdentifyObjectsInRegionEvaluator,
    "O-10_shape_outline_fill_data-generator": InDomainShapeOutlineFillEvaluator,
    "O-12_shape_color_then_scale_data-generator": InDomainShapeColorThenScaleEvaluator,
    "O-13_shape_outline_then_move_data-generator": InDomainShapeOutlineThenMoveEvaluator,
    "O-14_shape_scale_then_outline_data-generator": InDomainShapeScaleThenOutlineEvaluator,
    "O-15_ball_bounces_given_time_data-generator": InDomainBallBounceEvaluator,
    "O-16_color_addition_data-generator": InDomainColorAdditionEvaluator,
    "O-18_glass_refraction_data-generator": InDomainGlassRefractionEvaluator,
    "O-19_mirror_reflection_data-generator": InDomainMirrorReflectionEvaluator,
    "O-21_construction_blueprint_data-generator": InDomainConstructionBlueprintEvaluator,
    "O-23_domino_chain_branch_path_prediction_data-generator": InDomainDominoChainBranchEvaluator,
    "O-24_domino_chain_gap_analysis_data-generator": InDomainDominoChainGapEvaluator,
    "O-25_LEGO_construction_assembly_data-generator": InDomainLEGOConstructionEvaluator,
    "O-29_ballcolor_data-generator": InDomainBallColorEvaluator,
    "O-30_bookshelf_data-generator": InDomainBookshelfEvaluator,
    "O-31_ball_eating_data-generator": InDomainBallEatingEvaluator,
    "O-32_rolling_ball_data-generator": InDomainRollingBallEvaluator,
    "O-33_counting_object_data-generator": InDomainCountingObjectEvaluator,
    "O-34_dot_to_dot_task_data-generator": InDomainDotToDotEvaluator,
    "O-36_grid_shift_data-generator": InDomainGridShiftEvaluator,
    "O-37_light_sequence_data-generator": InDomainLightSequenceEvaluator,
    "O-38_majority_color_data-generator": InDomainMajorityColorEvaluator,
    "O-44_rotation_puzzle_data-generator": InDomainRotationPuzzleEvaluator,
    "O-45_sequence_completion_data-generator": InDomainSequenceCompletionEvaluator,
    "O-47_sliding_puzzle_data-generator": InDomainSlidingPuzzleEvaluator,
    "O-52_traffic_light_data-generator": InDomainTrafficLightEvaluator,
    "O-53_clock_data-generator": InDomainClockTimeEvaluator,
    "O-55_rotation_data-generator": InDomainRotationEvaluator,
    "O-75_communicating_vessels_data-generator": InDomainCommunicatingVesselsEvaluator,
    # Out-of-Domain_50 Tasks (50 tasks)
    "G-135_select_next_figure_small_large_alternating_sequence_data-generator": SelectNextFigureAlternatingEvaluator,
    "G-193_draw_next_sized_shape_data-generator": OutOfDomainDrawNextShapeEvaluator,
    "G-136_locate_point_in_overlapping_area_data-generator": LocatePointInOverlappingAreaEvaluator,
    "G-140_locate_topmost_unobscured_figure_data-generator": OutOfDomainLocateTopmostFigureEvaluator,
    "G-147_identify_unique_figure_in_uniform_set_data-generator": OutOfDomainIdentifyUniqueFigureEvaluator,
    "G-160_circle_largest_numerical_value_data-generator": OutOfDomainCircleLargestEvaluator,
    "G-161_mark_second_largest_shape_data-generator": OutOfDomainMarkSecondLargestEvaluator,
    "G-167_select_longest_polygon_side_data-generator": OutOfDomainSelectLongestSideEvaluator,
    "G-174_arrange_circles_by_circumference_data-generator": OutOfDomainArrangeCirclesEvaluator,
    "G-202_mark_wave_peaks_data-generator": MarkWavePeaksEvaluator,
    "G-212_find_incorrect_arrow_direction_data-generator": FindIncorrectArrowDirectionEvaluator,
    "G-217_circle_central_dot_data-generator": CircleCentralDotEvaluator,
    "G-218_identify_largest_angle_in_triangle_data-generator": IdentifyLargestAngleEvaluator,
    "G-219_select_leftmost_shape_data-generator": SelectLeftmostShapeEvaluator,
    "G-221_outline_innermost_square_data-generator": OutlineInnermostSquareEvaluator,
    "G-240_add_borders_to_unbordered_shapes_data-generator": AddBordersToUnborderedEvaluator,
    "G-247_identify_chinese_character_data-generator": IdentifyChineseCharacterEvaluator,
    "G-248_mark_asymmetrical_shape_data-generator": MarkAsymmetricalShapeEvaluator,
    "G-273_high_density_liquid_data-generator": HighDensityLiquidEvaluator,
    "G-47_multiple_keys_for_one_door_data-generator": MultipleKeysForOneDoorEvaluator,
    "O-2_pigment_color_mixing_subtractive_data-generator": OutOfDomainPigmentMixingEvaluator,
    "O-5_symbol_deletion_data-generator": OutOfDomainSymbolDeletionEvaluator,
    "O-6_2d_geometric_transformation_data-generator": OutOfDomainGeometricTransformEvaluator,
    "O-9_shape_scaling_data-generator": OutOfDomainShapeScalingEvaluator,
    "O-11_shape_color_then_move_data-generator": OutOfDomainShapeColorMoveEvaluator,
    "O-22_construction_stack_data-generator": OutOfDomainConstructionStackEvaluator,
    "O-39_maze_data-generator": OutOfDomainMazeEvaluator,
    "O-43_object_subtraction_data-generator": OutOfDomainObjectSubtractionEvaluator,
    "O-46_shape_sorter_data-generator": OutOfDomainShapeSorterEvaluator,
    "O-49_symmetry_completion_data-generator": OutOfDomainSymmetryEvaluator,
    "O-54_control_panel_data-generator": OutOfDomainControlPanelEvaluator,
    "O-56_raven_data-generator": OutOfDomainRavenMatrixEvaluator,
    "O-58_symbol_delete_data-generator": OutOfDomainSymbolDeleteEvaluator,
    "O-59_symbol_insert_data-generator": OutOfDomainSymbolInsertEvaluator,
    "O-60_symbol_substitute_data-generator": OutOfDomainSymbolSubstituteEvaluator,
    "O-61_symbol_edit_data-generator": OutOfDomainSymbolEditEvaluator,
    "O-62_gravity_physics_data-generator": OutOfDomainGravityPhysicsEvaluator,
    "O-64_animal_matching_data-generator": OutOfDomainAnimalMatchingEvaluator,
    "O-65_animal_size_sorting_data-generator": OutOfDomainAnimalSizeSortingEvaluator,
    "O-85_2d_object_rotation_data-generator": OutOfDomainObjectRotation2DEvaluator,
    # 10 additional tasks in Out-of-Domain_50
    "G-24_separate_objects_no_spin_data-generator": InDomainSeparateObjectsNoSpinEvaluator,
    "G-54_connecting_color_data-generator": InDomainConnectingColorEvaluator,
    "G-168_identify_nearest_to_square_rectangle_data-generator": InDomainIdentifyNearestSquareRectangleEvaluator,
    "G-169_locate_intersection_of_segments_data-generator": InDomainLocateSegmentIntersectionEvaluator,
    "G-189_draw_midpoint_perpendicular_line_data-generator": InDomainDrawMidpointPerpendicularEvaluator,
    "G-206_identify_pentagons_data-generator": InDomainIdentifyPentagonsEvaluator,
    "G-222_mark_tangent_point_of_circles_data-generator": InDomainMarkTangentPointEvaluator,
    "G-223_highlight_horizontal_lines_data-generator": InDomainHighlightHorizontalLinesEvaluator,
    "G-250_color_triple_intersection_red_data-generator": InDomainColorTripleIntersectionEvaluator,
    "O-27_move_2_object_to_2_target_data-generator": InDomainMoveObjectsToTargetEvaluator,
}


def get_evaluator(task_name: str, device: str = "cuda") -> BaseEvaluator:
    """Get the appropriate evaluator for a given task."""
    evaluator_class = TASK_EVALUATOR_MAP.get(task_name, BaseEvaluator)
    return evaluator_class(device=device, task_name=task_name)


def list_all_tasks():
    """List all 100 task names."""
    return list(TASK_EVALUATOR_MAP.keys())


# Task category mapping (cognitive categories for all 100 tasks)
TASK_CATEGORY_MAP = {
    # ===== In-Domain_50 Testset =====
    # Abstraction
    "O-10_shape_outline_fill_data-generator": "Abstraction",
    "O-12_shape_color_then_scale_data-generator": "Abstraction",
    "O-13_shape_outline_then_move_data-generator": "Abstraction",
    "O-14_shape_scale_then_outline_data-generator": "Abstraction",
    "O-21_construction_blueprint_data-generator": "Abstraction",
    "O-29_ballcolor_data-generator": "Abstraction",
    "O-30_bookshelf_data-generator": "Abstraction",
    "O-37_light_sequence_data-generator": "Abstraction",
    "O-45_sequence_completion_data-generator": "Abstraction",
    "O-47_sliding_puzzle_data-generator": "Abstraction",
    "G-29_chart_extreme_with_data_data-generator": "Abstraction",
    "G-41_grid_highest_cost_data-generator": "Abstraction",
    "G-51_predict_next_color_data-generator": "Abstraction",
    "G-131_select_next_figure_increasing_size_sequence_data-generator": "Abstraction",
    "G-134_select_next_figure_large_small_alternating_sequence_data-generator": "Abstraction",
    # Knowledge
    "O-15_ball_bounces_given_time_data-generator": "Knowledge",
    "O-18_glass_refraction_data-generator": "Knowledge",
    "O-19_mirror_reflection_data-generator": "Knowledge",
    "O-23_domino_chain_branch_path_prediction_data-generator": "Knowledge",
    "O-24_domino_chain_gap_analysis_data-generator": "Knowledge",
    "O-34_dot_to_dot_task_data-generator": "Knowledge",
    "O-52_traffic_light_data-generator": "Knowledge",
    "O-53_clock_data-generator": "Knowledge",
    "O-75_communicating_vessels_data-generator": "Knowledge",
    # Perception
    "O-16_color_addition_data-generator": "Perception",
    "O-31_ball_eating_data-generator": "Perception",
    "O-33_counting_object_data-generator": "Perception",
    "O-38_majority_color_data-generator": "Perception",
    "G-3_stable_sort_data-generator": "Perception",
    "G-5_multi_object_placement_data-generator": "Perception",
    "G-9_identify_objects_in_region_data-generator": "Perception",
    "G-39_attention_shift_different_data-generator": "Perception",
    "G-43_understand_scene_structure_data-generator": "Perception",
    "G-138_spot_unique_non_repeated_color_data-generator": "Perception",
    "G-158_identify_all_hollow_points_data-generator": "Perception",
    # Spatiality
    "O-25_LEGO_construction_assembly_data-generator": "Spatiality",
    "O-55_rotation_data-generator": "Spatiality",
    "G-13_grid_number_sequence_data-generator": "Spatiality",
    "G-15_grid_avoid_obstacles_data-generator": "Spatiality",
    "G-16_grid_go_through_block_data-generator": "Spatiality",
    "G-18_grid_shortest_path_data-generator": "Spatiality",
    "G-31_directed_graph_navigation_data-generator": "Spatiality",
    "G-45_key_door_matching_data-generator": "Spatiality",
    # Transformation
    "O-32_rolling_ball_data-generator": "Transformation",
    "O-36_grid_shift_data-generator": "Transformation",
    "O-44_rotation_puzzle_data-generator": "Transformation",
    "G-8_track_object_movement_data-generator": "Transformation",
    "G-21_multiple_occlusions_vertical_data-generator": "Transformation",
    "G-25_seperate_object_spinning_data-generator": "Transformation",
    "G-194_construct_concentric_ring_data-generator": "Transformation",
    # ===== Out-of-Domain_50 Testset =====
    # Abstraction
    "O-9_shape_scaling_data-generator": "Abstraction",
    "O-11_shape_color_then_move_data-generator": "Abstraction",
    "O-43_object_subtraction_data-generator": "Abstraction",
    "O-49_symmetry_completion_data-generator": "Abstraction",
    "O-54_control_panel_data-generator": "Abstraction",
    "O-56_raven_data-generator": "Abstraction",
    "G-135_select_next_figure_small_large_alternating_sequence_data-generator": "Abstraction",
    "G-193_draw_next_sized_shape_data-generator": "Abstraction",
    # Knowledge
    "O-62_gravity_physics_data-generator": "Knowledge",
    "G-160_circle_largest_numerical_value_data-generator": "Knowledge",
    "G-217_circle_central_dot_data-generator": "Knowledge",
    "G-247_identify_chinese_character_data-generator": "Knowledge",
    "G-273_high_density_liquid_data-generator": "Knowledge",
    # Perception
    "O-2_pigment_color_mixing_subtractive_data-generator": "Perception",
    "O-65_animal_size_sorting_data-generator": "Perception",
    "G-54_connecting_color_data-generator": "Perception",
    "G-136_locate_point_in_overlapping_area_data-generator": "Perception",
    "G-147_identify_unique_figure_in_uniform_set_data-generator": "Perception",
    "G-161_mark_second_largest_shape_data-generator": "Perception",
    "G-167_select_longest_polygon_side_data-generator": "Perception",
    "G-168_identify_nearest_to_square_rectangle_data-generator": "Perception",
    "G-169_locate_intersection_of_segments_data-generator": "Perception",
    "G-174_arrange_circles_by_circumference_data-generator": "Perception",
    "G-189_draw_midpoint_perpendicular_line_data-generator": "Perception",
    "G-202_mark_wave_peaks_data-generator": "Perception",
    "G-206_identify_pentagons_data-generator": "Perception",
    "G-212_find_incorrect_arrow_direction_data-generator": "Perception",
    "G-218_identify_largest_angle_in_triangle_data-generator": "Perception",
    "G-222_mark_tangent_point_of_circles_data-generator": "Perception",
    "G-223_highlight_horizontal_lines_data-generator": "Perception",
    "G-248_mark_asymmetrical_shape_data-generator": "Perception",
    "G-250_color_triple_intersection_red_data-generator": "Perception",
    # Spatiality
    "O-39_maze_data-generator": "Spatiality",
    "G-47_multiple_keys_for_one_door_data-generator": "Spatiality",
    "G-140_locate_topmost_unobscured_figure_data-generator": "Spatiality",
    "G-219_select_leftmost_shape_data-generator": "Spatiality",
    "G-221_outline_innermost_square_data-generator": "Spatiality",
    # Transformation
    "O-5_symbol_deletion_data-generator": "Transformation",
    "O-6_2d_geometric_transformation_data-generator": "Transformation",
    "O-22_construction_stack_data-generator": "Transformation",
    "O-27_move_2_object_to_2_target_data-generator": "Transformation",
    "O-46_shape_sorter_data-generator": "Transformation",
    "O-58_symbol_delete_data-generator": "Transformation",
    "O-59_symbol_insert_data-generator": "Transformation",
    "O-60_symbol_substitute_data-generator": "Transformation",
    "O-61_symbol_edit_data-generator": "Transformation",
    "O-64_animal_matching_data-generator": "Transformation",
    "O-85_2d_object_rotation_data-generator": "Transformation",
    "G-24_separate_objects_no_spin_data-generator": "Transformation",
    "G-240_add_borders_to_unbordered_shapes_data-generator": "Transformation",
}


def get_task_category(task_name: str) -> str:
    """Get the category for a task."""
    return TASK_CATEGORY_MAP.get(task_name, "Unknown")


def get_tasks_by_category():
    """Get tasks organized by category."""
    categories = {}
    for task_name in TASK_EVALUATOR_MAP.keys():
        category = get_task_category(task_name)
        if category not in categories:
            categories[category] = []
        categories[category].append(task_name)
    return categories


def get_tasks_by_category_and_split():
    """
    Get tasks organized by category, separated by split (In-Domain vs Out-of-Domain).

    Returns:
        dict: {
            'In_Domain': {category: [task_names], ...},
            'Out_of_Domain': {category: [task_names], ...}
        }
    """
    splits = get_tasks_by_split()

    result = {"In_Domain": {}, "Out_of_Domain": {}}

    for split_name, tasks in splits.items():
        for task_name in tasks:
            category = get_task_category(task_name)
            if category not in result[split_name]:
                result[split_name][category] = []
            result[split_name][category].append(task_name)

    return result


# Out-of-Domain_50 task prefixes (50 tasks total)
OUT_OF_DOMAIN_PREFIXES = [
    "G-135_",
    "G-193_",
    "G-136_",
    "G-140_",
    "G-147_",
    "G-160_",
    "G-161_",
    "G-167_",
    "G-202_",
    "G-212_",
    "G-217_",
    "G-218_",
    "G-219_",
    "G-221_",
    "G-240_",
    "G-247_",
    "G-248_",
    "G-174_",
    "G-273_",
    "G-47_",
    "O-11_",
    "O-56_",
    "O-22_",
    "O-2_",
    "O-39_",
    "O-43_",
    "O-46_",
    "O-49_",
    "O-5_",
    "O-54_",
    "O-58_",
    "O-59_",
    "O-60_",
    "O-61_",
    "O-62_",
    "O-64_",
    "O-65_",
    "O-6_",
    "O-85_",
    "O-9_",
    "G-24_",
    "G-54_",
    "G-168_",
    "G-169_",
    "G-189_",
    "G-206_",
    "G-222_",
    "G-223_",
    "G-250_",
    "O-27_",
]


def is_out_of_domain(task_name: str) -> bool:
    """Check if a task belongs to the Out-of-Domain split."""
    return any(task_name.startswith(p) for p in OUT_OF_DOMAIN_PREFIXES)


def get_split(task_name: str) -> str:
    """Get the split name ('Out_of_Domain' or 'In_Domain') for a task."""
    return "Out_of_Domain" if is_out_of_domain(task_name) else "In_Domain"


def get_tasks_by_split():
    """
    Get tasks organized by split (In-Domain_50 vs Out-of-Domain_50).

    Out-of-Domain_50 = 50 tasks
    In-Domain_50 = 50 tasks
    """
    out_of_domain = [t for t in TASK_EVALUATOR_MAP.keys() if is_out_of_domain(t)]
    in_domain = [t for t in TASK_EVALUATOR_MAP.keys() if not is_out_of_domain(t)]

    return {"In_Domain": in_domain, "Out_of_Domain": out_of_domain}


__all__ = [
    "BaseEvaluator",
    "get_evaluator",
    "list_all_tasks",
    "get_tasks_by_split",
    "get_tasks_by_category",
    "get_tasks_by_category_and_split",
    "get_task_category",
    "is_out_of_domain",
    "get_split",
    "TASK_EVALUATOR_MAP",
    "TASK_CATEGORY_MAP",
    "OUT_OF_DOMAIN_PREFIXES",
]
