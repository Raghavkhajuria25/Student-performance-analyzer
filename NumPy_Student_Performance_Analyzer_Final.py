import numpy as np

np.random.seed(42)

student_ids = np.arange(1, 51)
age = np.random.randint(18, 26, 50)
study_hrs = np.random.randint(1, 11, 50)
attendence = np.random.randint(50, 101, 50)
assignment_score = np.random.randint(0, 101, 50)
midterm_score = np.random.randint(0, 101, 50)
final_score = np.random.randint(0, 101, 50)

data = np.column_stack((
    student_ids,
    age,
    study_hrs,
    attendence,
    assignment_score,
    midterm_score,
    final_score
))

high_attendence = data[data[:, 3] > 75]

condition = (data[:, 2] >= 5) & (data[:, 3] >= 75)

filtered_students = data[condition]


print("Number of Students:", len(filtered_students))

average_study_hours = np.mean(data[:, 2])

average_attendence = np.mean(data[:, 3])

scores = data[:, 4:7]

average_scores = np.mean(scores, axis=0)

median_score = np.median(scores, axis=0)

score_std = np.std(scores, axis=0)

score_variance = np.var(scores, axis=0)

min_scores = np.min(scores, axis=0)

max_scores = np.max(scores, axis=0)

print("\n--- Highest and Lowest Final Score ---")

final_scores = data[:, 6]

highest_index = np.argmax(final_scores)
lowest_index = np.argmin(final_scores)

highest_student_id = data[highest_index, 0]
lowest_student_id = data[lowest_index, 0]

print("Highest Final Score:", final_scores[highest_index])
print("Highest Final Score Student ID:", highest_student_id)

print("Lowest Final Score:", final_scores[lowest_index])
print("Lowest Final Score Student ID:", lowest_student_id)

result = np.where(
    final_scores >= 40,
    "Pass",
    "Fail"
)

pass_count = np.sum(result == "Pass")
fail_count = np.sum(result == "Fail")

performance = np.where(
    final_scores >= 80,
    "Excellent",
    np.where(
        final_scores >= 60,
        "Good",
        np.where(
            final_scores >= 40,
            "Average",
            "Needs Improvement"
        )
    )
)

unique_categories, category_count = np.unique(
    performance,
    return_counts=True
)

ranking_order = np.argsort(final_scores)

ranking_order = ranking_order[::-1]

top_5_indices = ranking_order[:5]

sorted_final_scores = np.sort(final_scores)

highest_scores = sorted_final_scores[::-1]

overall_scores = np.zeros(50)

overall_scores = (
    scores[:, 0] * 0.20
    + scores[:, 1] * 0.30
    + scores[:, 2] * 0.50
)

bonus = np.ones(50)

bonus_marks = np.full(50, 5)

attendence_bonus = np.where(
    attendence >= 90,
    5,
    0
)

overall_scores = (
    scores[:, 0] * 0.20
    + scores[:, 1] * 0.30
    + scores[:, 2] * 0.50
    + attendence_bonus
)

data_float = data.astype(float)

data_float[4, 6] = np.nan

average_final = np.nanmean(data_float[:, 6])

median_final = np.nanmedian(data_float[:, 6])

std_final = np.nanstd(data_float[:, 6])

min_final = np.nanmin(data_float[:, 6])
max_final = np.nanmax(data_float[:, 6])

final_score_matrix = final_scores.reshape(5, 10)

transposed_scores = final_score_matrix.T

flat_scores = final_score_matrix.flatten()

ravel_scores = final_score_matrix.ravel()

grace_marks = 2

adjusted_scores = scores + grace_marks

grace_marks = np.array([2, 3, 5])

adjusted_scores = scores + grace_marks

new_student_data = np.column_stack((
    np.arange(51, 56),
    np.random.randint(18, 26, 5),
    np.random.randint(1, 11, 5),
    np.random.randint(50, 101, 5),
    np.random.randint(0, 101, 5),
    np.random.randint(0, 101, 5),
    np.random.randint(0, 101, 5)
))

updated_data = np.concatenate(
    (data, new_student_data),
    axis=0
)

stacked_data = np.vstack(
    (data, new_student_data)
)

overall_column = overall_scores.reshape(50, 1)

data_with_overall = np.hstack(
    (data, overall_column)
)

student_groups = np.split(
    data,
    5,
    axis=0
)

study_efficiency = np.random.rand(50) * 100

performance_variation = np.random.randn(50)

sections = np.random.choice(
    ["A", "B", "C"],
    size=50
)

weights = np.array([0.20, 0.30, 0.50])

weighted_scores = np.dot(
    scores,
    weights
)

weighted_scores_at = scores @ weights

weighted_scores_matmul = np.matmul(
    scores,
    weights
)

min_score = np.min(weighted_scores)
max_score = np.max(weighted_scores)

normalized_scores = (
    (weighted_scores - min_score)
    / (max_score - min_score)
)

normalized_scores_100 = normalized_scores * 100

study_efficiency_column = study_efficiency.reshape(
    50,
    1
)

normalized_score_column = normalized_scores_100.reshape(
    50,
    1
)

final_analysis_data = np.hstack((
    data,
    study_efficiency_column,
    normalized_score_column
))

total_students = final_analysis_data.shape[0]

print("\n========== STUDENT PERFORMANCE REPORT ==========")

print("Total Students:", total_students)

print("Average Study Hours:", np.mean(data[:, 2]))
print("Average Attendance:", np.mean(data[:, 3]))
print("Average Assignment Score:", np.mean(data[:, 4]))
print("Average Midterm Score:", np.mean(data[:, 5]))
print("Average Final Score:", np.mean(data[:, 6]))
print("Average Overall Score:", np.mean(overall_scores))

print("\n--- Pass / Fail Summary ---")

print("Passed Students:", pass_count)
print("Failed Students:", fail_count)

pass_percentage = (
    pass_count / total_students
) * 100

print("Pass Percentage:", round(pass_percentage, 2), "%")

print("\n--- Performance Category Summary ---")

for i in range(len(unique_categories)):
    print(
        unique_categories[i],
        ":",
        category_count[i]
    )

overall_ranking = np.argsort(
    overall_scores
)[::-1]

top_5_overall = overall_ranking[:5]

top_5_ids = data[
    top_5_overall,
    0
]

top_5_scores = overall_scores[
    top_5_overall
]

print("\n--- Top 5 Students ---")

for i in range(5):
    print(
        "Rank",
        i + 1,
        "| Student ID:",
        int(top_5_ids[i]),
        "| Overall Score:",
        round(top_5_scores[i], 2)
    )

high_attendance_count = np.sum(
    attendence >= 90
)

print("\n--- Attendance Analysis ---")

print(
    "Students with 90%+ Attendance:",
    high_attendance_count
)

high_attendance_percentage = (
    high_attendance_count / total_students
) * 100

print(
    "90%+ Attendance Percentage:",
    high_attendance_percentage
)

high_study_count = np.sum(
    study_hrs >= 5
)

print("\n--- Study Hours Analysis ---")

print(
    "Students studying 5+ hours:",
    high_study_count
)

high_study_percentage = (
    high_study_count / total_students
) * 100

print(
    "5+ Hours Study Percentage:",
    high_study_percentage
)

lowest_5_overall = np.argsort(
    overall_scores
)[:5]

lowest_5_ids = data[
    lowest_5_overall,
    0
]

lowest_5_scores = overall_scores[
    lowest_5_overall
]

print("\n--- Lowest 5 Students ---")

for i in range(5):
    print(
        "Rank",
        i + 1,
        "| Student ID:",
        int(lowest_5_ids[i]),
        "| Overall Score:",
        round(lowest_5_scores[i], 2)
    )

group_averages = np.zeros(5)

for i in range(5):
    group_averages[i] = np.mean(
        student_groups[i][:, 6]
    )

print("\n--- Group-wise Performance ---")

for i in range(len(group_averages)):
    print(
        "Group",
        i + 1,
        "| Average Final Score:",
        round(group_averages[i], 2)
    )

score_scale = np.linspace(
    0,
    100,
    11
)

print("\n--- Score Range Analysis ---")

below_40 = np.sum(
    final_scores < 40
)

score_40_59 = np.sum(
    (final_scores >= 40)
    & (final_scores < 60)
)

score_60_79 = np.sum(
    (final_scores >= 60)
    & (final_scores < 80)
)

score_80_plus = np.sum(
    final_scores >= 80
)

print("Below 40:", below_40)
print("40 - 59:", score_40_59)
print("60 - 79:", score_60_79)
print("80+:", score_80_plus)

print("\n========== FINAL SUMMARY ==========")

print("Total Students:", total_students)

print(
    "Average Final Score:",
    round(np.mean(final_scores), 2)
)

print(
    "Average Overall Score:",
    round(np.mean(overall_scores), 2)
)

print(
    "Pass Percentage:",
    round(pass_percentage, 2),
    "%"
)

print(
    "90%+ Attendance:",
    high_attendance_count
)

print(
    "5+ Hours Study:",
    high_study_count
)

print("\n========== PROJECT COMPLETED ==========")
