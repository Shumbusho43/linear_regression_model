import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:student_performance_app/main.dart';

void main() {
  testWidgets('App renders the prediction form and Predict button',
      (WidgetTester tester) async {
    await tester.pumpWidget(const StudentPredictorApp());

    // The Predict button is present.
    expect(find.text('Predict'), findsOneWidget);

    // One text field per model input (8 variables).
    expect(find.byType(TextFormField), findsNWidgets(8));
  });
}
