import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

// ---------------------------------------------------------------------------
// CONFIGURATION
// Replace kApiBaseUrl with the base URL of your deployed FastAPI service
// from Task 2. No trailing slash.
// Example: https://student-performance-api.onrender.com
// ---------------------------------------------------------------------------
const String kApiBaseUrl = 'https://student-performance-api.onrender.com';
const String kPredictPath = '/predict';

void main() {
  runApp(const StudentPredictorApp());
}

class StudentPredictorApp extends StatelessWidget {
  const StudentPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Student Performance Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0F5C6B)),
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          isDense: true,
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

// ---------------------------------------------------------------------------
// Input field definition: one per model variable, with its valid range.
// The ranges mirror the Pydantic constraints enforced by the API.
// ---------------------------------------------------------------------------
class InputField {
  InputField({
    required this.name,
    required this.label,
    required this.helper,
    required this.min,
    required this.max,
    required this.icon,
    this.isInteger = true,
  });

  final String name;
  final String label;
  final String helper;
  final num min;
  final num max;
  final IconData icon;
  final bool isInteger;
  final TextEditingController controller = TextEditingController();
}

class PredictionResult {
  const PredictionResult({
    required this.grade,
    required this.percentage,
    required this.riskLevel,
    required this.interpretation,
  });

  final double grade;
  final double percentage;
  final String riskLevel;
  final String interpretation;

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      grade: (json['predicted_grade'] as num).toDouble(),
      percentage: (json['grade_percentage'] as num?)?.toDouble() ?? 0,
      riskLevel: json['risk_level'] as String? ?? 'Unknown',
      interpretation: json['interpretation'] as String? ?? '',
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();

  // Eight fields, one for each variable the model needs.
  final List<InputField> _fields = [
    InputField(
      name: 'G1',
      label: 'First-period grade (G1)',
      helper: '0 to 20',
      min: 0,
      max: 20,
      icon: Icons.grade_outlined,
      isInteger: false,
    ),
    InputField(
      name: 'failures',
      label: 'Past class failures',
      helper: '0 to 4',
      min: 0,
      max: 4,
      icon: Icons.replay_outlined,
    ),
    InputField(
      name: 'Medu',
      label: "Mother's education level",
      helper: '0 = none, 4 = higher education',
      min: 0,
      max: 4,
      icon: Icons.school_outlined,
    ),
    InputField(
      name: 'studytime',
      label: 'Weekly study time',
      helper: '1 = under 2h, 4 = over 10h',
      min: 1,
      max: 4,
      icon: Icons.timer_outlined,
    ),
    InputField(
      name: 'absences',
      label: 'School absences',
      helper: '0 to 93',
      min: 0,
      max: 93,
      icon: Icons.event_busy_outlined,
    ),
    InputField(
      name: 'internet',
      label: 'Internet access at home',
      helper: '0 = no, 1 = yes',
      min: 0,
      max: 1,
      icon: Icons.wifi_outlined,
    ),
    InputField(
      name: 'higher',
      label: 'Wants higher education',
      helper: '0 = no, 1 = yes',
      min: 0,
      max: 1,
      icon: Icons.trending_up_outlined,
    ),
    InputField(
      name: 'age',
      label: 'Age',
      helper: '15 to 22',
      min: 15,
      max: 22,
      icon: Icons.cake_outlined,
    ),
  ];

  bool _loading = false;
  String? _errorMessage;
  PredictionResult? _result;

  @override
  void dispose() {
    for (final field in _fields) {
      field.controller.dispose();
    }
    super.dispose();
  }

  // -------------------------------------------------------------------------
  // Client-side validation: catches missing values, non-numeric input and
  // out-of-range values before a request is ever sent.
  // -------------------------------------------------------------------------
  String? _validateField(InputField field, String? value) {
    final raw = (value ?? '').trim();
    if (raw.isEmpty) {
      return 'This value is required';
    }
    final parsed = num.tryParse(raw);
    if (parsed == null) {
      return 'Enter a valid number';
    }
    if (field.isInteger && parsed % 1 != 0) {
      return 'Whole numbers only';
    }
    if (parsed < field.min || parsed > field.max) {
      return 'Must be between ${field.min} and ${field.max}';
    }
    return null;
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();

    if (!(_formKey.currentState?.validate() ?? false)) {
      setState(() {
        _result = null;
        _errorMessage =
            'Some values are missing or out of range. Please correct the '
            'highlighted fields and try again.';
      });
      return;
    }

    setState(() {
      _loading = true;
      _errorMessage = null;
      _result = null;
    });

    try {
      final payload = <String, dynamic>{};
      for (final field in _fields) {
        final value = num.parse(field.controller.text.trim());
        payload[field.name] =
            field.isInteger ? value.toInt() : value.toDouble();
      }

      final response = await http
          .post(
            Uri.parse('$kApiBaseUrl$kPredictPath'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 90));

      if (!mounted) return;

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() => _result = PredictionResult.fromJson(decoded));
      } else if (response.statusCode == 422) {
        setState(
            () => _errorMessage = _formatApiValidationError(response.body));
      } else {
        setState(() => _errorMessage =
            'The server returned status ${response.statusCode}. Please try again.');
      }
    } on TimeoutException {
      if (!mounted) return;
      setState(() => _errorMessage =
          'The request timed out. Free hosting puts the API to sleep when idle. '
          'Open the API URL in a browser once to wake it, then try again.');
    } catch (error) {
      if (!mounted) return;
      setState(() => _errorMessage =
          'Could not reach the prediction service. Check your internet '
          'connection and the API URL.\n\n$error');
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  /// Turns FastAPI's 422 response into a readable message.
  String _formatApiValidationError(String body) {
    try {
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      final detail = decoded['detail'];
      if (detail is List) {
        final lines = detail.map((item) {
          final map = item as Map<String, dynamic>;
          final location = (map['loc'] as List)
              .where((part) => part != 'body')
              .join('.');
          return '\u2022 $location: ${map['msg']}';
        }).join('\n');
        return 'The API rejected these values:\n$lines';
      }
      return 'The API rejected the request: $detail';
    } catch (_) {
      return 'The API rejected the request (status 422).';
    }
  }

  void _loadExample() {
    const example = {
      'G1': '8',
      'failures': '1',
      'Medu': '1',
      'studytime': '1',
      'absences': '6',
      'internet': '0',
      'higher': '1',
      'age': '17',
    };
    for (final field in _fields) {
      field.controller.text = example[field.name] ?? '';
    }
    setState(() {
      _result = null;
      _errorMessage = null;
    });
  }

  void _clearAll() {
    for (final field in _fields) {
      field.controller.clear();
    }
    _formKey.currentState?.reset();
    setState(() {
      _result = null;
      _errorMessage = null;
    });
  }

  Color _riskColor(String risk) {
    switch (risk) {
      case 'High risk':
        return const Color(0xFFC62828);
      case 'Moderate risk':
        return const Color(0xFFEF6C00);
      case 'On track':
        return const Color(0xFF2E7D32);
      default:
        return const Color(0xFF455A64);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      backgroundColor: const Color(0xFFF6F8F9),
      appBar: AppBar(
        title: const Text('Student Performance Predictor'),
        backgroundColor: theme.colorScheme.primary,
        foregroundColor: theme.colorScheme.onPrimary,
        actions: [
          IconButton(
            tooltip: 'Clear all fields',
            onPressed: _loading ? null : _clearAll,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildIntroCard(theme),
                    const SizedBox(height: 18),
                    Text(
                      'Student details',
                      style: theme.textTheme.titleMedium
                          ?.copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 12),
                    ..._fields.map(_buildTextField),
                    const SizedBox(height: 4),
                    Align(
                      alignment: Alignment.centerLeft,
                      child: TextButton.icon(
                        onPressed: _loading ? null : _loadExample,
                        icon: const Icon(Icons.auto_fix_high_outlined,
                            size: 18),
                        label: const Text('Load example values'),
                      ),
                    ),
                    const SizedBox(height: 12),
                    SizedBox(
                      height: 52,
                      child: FilledButton.icon(
                        onPressed: _loading ? null : _predict,
                        icon: _loading
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              )
                            : const Icon(Icons.insights),
                        label: Text(
                          _loading ? 'Predicting...' : 'Predict',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    _buildDisplayArea(theme),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildIntroCard(ThemeData theme) {
    return Card(
      elevation: 0,
      color: theme.colorScheme.primaryContainer,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.lightbulb_outline,
                color: theme.colorScheme.onPrimaryContainer),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Predict a student\u2019s final grade from socioeconomic and '
                'information-access factors, so that at-risk students can be '
                'connected to scholarships and support early.',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onPrimaryContainer,
                  height: 1.35,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextField(InputField field) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextFormField(
        controller: field.controller,
        keyboardType:
            TextInputType.numberWithOptions(decimal: !field.isInteger),
        inputFormatters: [
          FilteringTextInputFormatter.allow(
            RegExp(field.isInteger ? r'[0-9]' : r'[0-9.]'),
          ),
        ],
        textInputAction: TextInputAction.next,
        autovalidateMode: AutovalidateMode.onUserInteraction,
        decoration: InputDecoration(
          labelText: field.label,
          helperText: field.helper,
          prefixIcon: Icon(field.icon, size: 20),
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        ),
        validator: (value) => _validateField(field, value),
      ),
    );
  }

  /// Display area: shows the prediction, an error message, or a placeholder.
  Widget _buildDisplayArea(ThemeData theme) {
    if (_loading) {
      return _shellCard(
        color: const Color(0xFFECEFF1),
        child: const Padding(
          padding: EdgeInsets.symmetric(vertical: 12),
          child: Row(
            children: [
              SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 14),
              Expanded(child: Text('Contacting the prediction model...')),
            ],
          ),
        ),
      );
    }

    if (_errorMessage != null) {
      return _shellCard(
        color: const Color(0xFFFDECEA),
        border: const Color(0xFFE57373),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline, color: Color(0xFFC62828)),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _errorMessage!,
                style: const TextStyle(color: Color(0xFF8E1B18), height: 1.4),
              ),
            ),
          ],
        ),
      );
    }

    final result = _result;
    if (result == null) {
      return _shellCard(
        color: const Color(0xFFECEFF1),
        child: Row(
          children: [
            Icon(Icons.info_outline, color: theme.colorScheme.outline),
            const SizedBox(width: 12),
            const Expanded(
              child: Text(
                'Fill in all eight values and tap Predict to see the '
                'estimated final grade.',
              ),
            ),
          ],
        ),
      );
    }

    final color = _riskColor(result.riskLevel);
    return _shellCard(
      color: _tint(color, 0.08),
      border: _tint(color, 0.45),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'PREDICTED FINAL GRADE',
            style: theme.textTheme.labelMedium?.copyWith(
              letterSpacing: 1.1,
              color: theme.colorScheme.outline,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                result.grade.toStringAsFixed(2),
                style: theme.textTheme.displaySmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                '/ 20',
                style: theme.textTheme.titleMedium
                    ?.copyWith(color: theme.colorScheme.outline),
              ),
              const Spacer(),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  result.riskLevel,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: (result.grade / 20).clamp(0.0, 1.0),
              minHeight: 8,
              backgroundColor: _tint(color, 0.15),
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '${result.percentage.toStringAsFixed(1)}% of the maximum score',
            style: theme.textTheme.bodySmall
                ?.copyWith(color: theme.colorScheme.outline),
          ),
          if (result.interpretation.isNotEmpty) ...[
            const SizedBox(height: 14),
            const Divider(height: 1),
            const SizedBox(height: 14),
            Text(
              result.interpretation,
              style: theme.textTheme.bodyMedium?.copyWith(height: 1.4),
            ),
          ],
        ],
      ),
    );
  }

  /// Blend a colour toward white by [amount] (0..1). Avoids version-specific
  /// colour APIs so the app builds on a wide range of Flutter releases.
  Color _tint(Color base, double amount) {
    return Color.alphaBlend(base.withOpacity(amount), Colors.white);
  }

  Widget _shellCard({
    required Widget child,
    required Color color,
    Color? border,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: border ?? Colors.transparent,
          width: border == null ? 0 : 1.2,
        ),
      ),
      child: child,
    );
  }
}
