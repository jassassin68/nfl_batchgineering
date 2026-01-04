-- models/marts/mart_model_validation.sql
-- Backtesting framework for validating spread and totals predictions

{{ config(
    materialized='table'
) }}

WITH game_results AS (
    SELECT 
        game_id,
        season,
        week,
        home_team,
        away_team,
        home_score,
        away_score,
        (home_score - away_score) AS actual_spread,  -- Positive = home team won
        (home_score + away_score) AS actual_total
    FROM {{ source('nfl_data', 'schedule') }}
    WHERE home_score IS NOT NULL  -- Only completed games
      AND away_score IS NOT NULL
),

predictions_with_results AS (
    SELECT 
        gpf.*,
        gr.home_score,
        gr.away_score,
        gr.actual_spread,
        gr.actual_total,
        
        -- SPREAD PREDICTIONS
        
        -- Simple EPA-based spread prediction
        gpf.epa_differential * 14 + gpf.estimated_home_field_advantage AS epa_spread_prediction,
        
        -- Enhanced spread prediction incorporating multiple factors
        (gpf.epa_differential * 12) + 
        (gpf.success_rate_differential * 8) + 
        (gpf.matchup_advantage * 0.5) +
        gpf.estimated_home_field_advantage +
        gpf.weather_totals_adjustment * 0.3 AS enhanced_spread_prediction,
        
        -- TOTALS PREDICTIONS
        
        -- Base total from offensive efficiency
        42 + (gpf.combined_offensive_epa * 25) - (gpf.combined_defensive_epa * 15) AS base_total_prediction,
        
        -- Weather-adjusted total
        42 + (gpf.combined_offensive_epa * 25) - (gpf.combined_defensive_epa * 15) + 
        gpf.weather_totals_adjustment AS weather_adj_total_prediction,
        
        -- Enhanced total with situational factors
        42 + 
        (gpf.combined_offensive_epa * 20) - 
        (gpf.combined_defensive_epa * 12) +
        (gpf.combined_explosive_rate * 15) +
        (gpf.combined_rz_efficiency * 8) +
        gpf.weather_totals_adjustment +
        CASE WHEN gpf.is_division_game = 1 THEN -2.5 ELSE 0 END AS enhanced_total_prediction
        
    FROM {{ ref('mart_game_prediction_features') }} gpf
    INNER JOIN game_results gr
        ON gpf.game_id = gr.game_id
),

prediction_accuracy AS (
    SELECT 
        *,
        
        -- SPREAD PREDICTION ACCURACY
        
        -- Spread prediction errors
        ABS(epa_spread_prediction - actual_spread) AS epa_spread_error,
        ABS(enhanced_spread_prediction - actual_spread) AS enhanced_spread_error,
        
        -- Spread betting accuracy (did we pick the right side?)
        CASE 
            WHEN epa_spread_prediction > 0 AND actual_spread > 0 THEN 1  -- Predicted home win, home won
            WHEN epa_spread_prediction < 0 AND actual_spread < 0 THEN 1  -- Predicted away win, away won
            ELSE 0
        END AS epa_spread_correct,
        
        CASE 
            WHEN enhanced_spread_prediction > 0 AND actual_spread > 0 THEN 1
            WHEN enhanced_spread_prediction < 0 AND actual_spread < 0 THEN 1
            ELSE 0
        END AS enhanced_spread_correct,
        
        -- TOTALS PREDICTION ACCURACY
        
        -- Total prediction errors  
        ABS(base_total_prediction - actual_total) AS base_total_error,
        ABS(weather_adj_total_prediction - actual_total) AS weather_total_error,
        ABS(enhanced_total_prediction - actual_total) AS enhanced_total_error,
        
        -- Over/Under accuracy
        CASE 
            WHEN base_total_prediction > 47 AND actual_total > 47 THEN 1  -- Predicted over, actual over
            WHEN base_total_prediction < 47 AND actual_total < 47 THEN 1  -- Predicted under, actual under
            ELSE 0
        END AS base_total_correct,
        
        CASE 
            WHEN enhanced_total_prediction > 47 AND actual_total > 47 THEN 1
            WHEN enhanced_total_prediction < 47 AND actual_total < 47 THEN 1  
            ELSE 0
        END AS enhanced_total_correct,
        
        -- Confidence levels based on prediction strength
        CASE 
            WHEN ABS(epa_spread_prediction) >= 7 THEN 'High'
            WHEN ABS(epa_spread_prediction) >= 3 THEN 'Medium'  
            ELSE 'Low'
        END AS spread_confidence,
        
        CASE 
            WHEN ABS(enhanced_total_prediction - 47) >= 6 THEN 'High'
            WHEN ABS(enhanced_total_prediction - 47) >= 3 THEN 'Medium'
            ELSE 'Low'  
        END AS total_confidence
        
    FROM predictions_with_results
),

-- Model performance summary by season and confidence level
model_performance AS (
    SELECT 
        season,
        
        -- Overall counts
        COUNT(*) AS total_games,
        
        -- SPREAD PERFORMANCE
        AVG(epa_spread_correct::float) AS epa_spread_accuracy,
        AVG(enhanced_spread_correct::float) AS enhanced_spread_accuracy,
        AVG(epa_spread_error) AS avg_epa_spread_error,
        AVG(enhanced_spread_error) AS avg_enhanced_spread_error,
        
        -- TOTALS PERFORMANCE  
        AVG(base_total_correct::float) AS base_total_accuracy,
        AVG(enhanced_total_correct::float) AS enhanced_total_accuracy,
        AVG(base_total_error) AS avg_base_total_error,
        AVG(enhanced_total_error) AS avg_enhanced_total_error,
        
        -- Performance by confidence level - SPREAD
        AVG(CASE WHEN spread_confidence = 'High' THEN enhanced_spread_correct::float END) AS high_conf_spread_acc,
        COUNT(CASE WHEN spread_confidence = 'High' THEN 1 END) AS high_conf_spread_games,
        AVG(CASE WHEN spread_confidence = 'Medium' THEN enhanced_spread_correct::float END) AS med_conf_spread_acc,
        COUNT(CASE WHEN spread_confidence = 'Medium' THEN 1 END) AS med_conf_spread_games,
        
        -- Performance by confidence level - TOTALS
        AVG(CASE WHEN total_confidence = 'High' THEN enhanced_total_correct::float END) AS high_conf_total_acc,
        COUNT(CASE WHEN total_confidence = 'High' THEN 1 END) AS high_conf_total_games,
        AVG(CASE WHEN total_confidence = 'Medium' THEN enhanced_total_correct::float END) AS med_conf_total_acc,
        COUNT(CASE WHEN total_confidence = 'Medium' THEN 1 END) AS med_conf_total_games,
        
        -- Weather impact on predictions
        AVG(CASE WHEN wind >= 15 THEN enhanced_total_correct::float END) AS high_wind_total_acc,
        AVG(CASE WHEN temp <= 32 THEN enhanced_total_correct::float END) AS cold_weather_total_acc,
        
        -- Division game performance
        AVG(CASE WHEN is_division_game = 1 THEN enhanced_spread_correct::float END) AS division_spread_acc,
        AVG(CASE WHEN is_division_game = 1 THEN enhanced_total_correct::float END) AS division_total_acc,
        
        -- Close game performance
        AVG(CASE WHEN close_matchup = 1 THEN enhanced_spread_correct::float END) AS close_game_spread_acc,
        
        -- Model calibration (are high-confidence picks actually more accurate?)
        CASE 
            WHEN AVG(CASE WHEN spread_confidence = 'High' THEN enhanced_spread_correct::float END) > 
                 AVG(CASE WHEN spread_confidence = 'Low' THEN enhanced_spread_correct::float END)
            THEN 'Well_Calibrated'
            ELSE 'Poorly_Calibrated'
        END AS spread_calibration,
        
        CASE 
            WHEN AVG(CASE WHEN total_confidence = 'High' THEN enhanced_total_correct::float END) >
                 AVG(CASE WHEN total_confidence = 'Low' THEN enhanced_total_correct::float END)
            THEN 'Well_Calibrated' 
            ELSE 'Poorly_Calibrated'
        END AS total_calibration
        
    FROM prediction_accuracy
    GROUP BY season
),

-- Weekly performance trends
weekly_performance AS (
    SELECT 
        season,
        week,
        COUNT(*) AS games,
        AVG(enhanced_spread_correct::float) AS weekly_spread_acc,
        AVG(enhanced_total_correct::float) AS weekly_total_acc,
        AVG(enhanced_spread_error) AS weekly_spread_error,
        AVG(enhanced_total_error) AS weekly_total_error,
        
        -- Rolling 4-week accuracy
        AVG(AVG(enhanced_spread_correct::float)) OVER (
            PARTITION BY season 
            ORDER BY week 
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_spread_acc,
        
        AVG(AVG(enhanced_total_correct::float)) OVER (
            PARTITION BY season
            ORDER BY week
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW  
        ) AS rolling_total_acc
        
    FROM prediction_accuracy
    GROUP BY season, week
),

-- Feature importance analysis (which factors drive accuracy?)
feature_performance AS (
    SELECT 
        season,
        
        -- Performance when key features are strong
        AVG(CASE WHEN ABS(epa_differential) >= 0.15 THEN enhanced_spread_correct::float END) AS large_epa_diff_acc,
        AVG(CASE WHEN ABS(epa_differential) < 0.05 THEN enhanced_spread_correct::float END) AS small_epa_diff_acc,
        
        AVG(CASE WHEN combined_offensive_epa >= 0.2 THEN enhanced_total_correct::float END) AS high_offense_total_acc,
        AVG(CASE WHEN combined_offensive_epa <= -0.1 THEN enhanced_total_correct::float END) AS low_offense_total_acc,
        
        -- Weather impact validation  
        AVG(CASE WHEN ABS(weather_totals_adjustment) >= 2 THEN enhanced_total_correct::float END) AS weather_game_acc,
        AVG(CASE WHEN weather_totals_adjustment = 0 THEN enhanced_total_correct::float END) AS good_weather_acc,
        
        -- Matchup quality impact
        AVG(CASE WHEN matchup_advantage >= 2 THEN enhanced_spread_correct::float END) AS strong_matchup_acc,
        AVG(CASE WHEN ABS(matchup_advantage) < 0.5 THEN enhanced_spread_correct::float END) AS even_matchup_acc
        
    FROM prediction_accuracy  
    GROUP BY season
),

-- ROI Analysis (assuming standard -110 betting lines)
betting_roi AS (
    SELECT 
        season,
        
        -- Spread betting ROI (need >52.38% to break even at -110)
        COUNT(CASE WHEN enhanced_spread_correct = 1 THEN 1 END) AS spread_wins,
        COUNT(*) AS total_spread_bets,
        AVG(enhanced_spread_correct::float) AS spread_win_rate,
        
        -- Calculate ROI assuming $100 bets at -110 odds
        CASE 
            WHEN AVG(enhanced_spread_correct::float) > 0.5238 THEN
                (COUNT(CASE WHEN enhanced_spread_correct = 1 THEN 1 END) * 90.91) - 
                (COUNT(CASE WHEN enhanced_spread_correct = 0 THEN 1 END) * 100)
            ELSE 
                (COUNT(CASE WHEN enhanced_spread_correct = 1 THEN 1 END) * 90.91) - 
                (COUNT(CASE WHEN enhanced_spread_correct = 0 THEN 1 END) * 100)
        END AS spread_profit_per_100_games,
        
        -- Totals betting ROI
        COUNT(CASE WHEN enhanced_total_correct = 1 THEN 1 END) AS total_wins,
        COUNT(*) AS total_total_bets,
        AVG(enhanced_total_correct::float) AS total_win_rate,
        
        CASE 
            WHEN AVG(enhanced_total_correct::float) > 0.5238 THEN
                (COUNT(CASE WHEN enhanced_total_correct = 1 THEN 1 END) * 90.91) - 
                (COUNT(CASE WHEN enhanced_total_correct = 0 THEN 1 END) * 100)
            ELSE
                (COUNT(CASE WHEN enhanced_total_correct = 1 THEN 1 END) * 90.91) - 
                (COUNT(CASE WHEN enhanced_total_correct = 0 THEN 1 END) * 100)
        END AS total_profit_per_100_games,
        
        -- High confidence betting only
        AVG(CASE WHEN spread_confidence = 'High' THEN enhanced_spread_correct::float END) AS high_conf_spread_rate,
        AVG(CASE WHEN total_confidence = 'High' THEN enhanced_total_correct::float END) AS high_conf_total_rate,
        
        COUNT(CASE WHEN spread_confidence = 'High' THEN 1 END) AS high_conf_spread_opportunities,
        COUNT(CASE WHEN total_confidence = 'High' THEN 1 END) AS high_conf_total_opportunities
        
    FROM prediction_accuracy
    GROUP BY season
)

-- Final comprehensive validation report
SELECT 
    mp.season,
    mp.total_games,
    
    -- SPREAD PREDICTION PERFORMANCE
    ROUND(mp.epa_spread_accuracy * 100, 1) AS epa_spread_accuracy_pct,
    ROUND(mp.enhanced_spread_accuracy * 100, 1) AS enhanced_spread_accuracy_pct,
    ROUND(mp.avg_enhanced_spread_error, 2) AS avg_spread_error_points,
    
    -- TOTALS PREDICTION PERFORMANCE  
    ROUND(mp.base_total_accuracy * 100, 1) AS base_total_accuracy_pct,
    ROUND(mp.enhanced_total_accuracy * 100, 1) AS enhanced_total_accuracy_pct,
    ROUND(mp.avg_enhanced_total_error, 2) AS avg_total_error_points,
    
    -- CONFIDENCE CALIBRATION
    ROUND(mp.high_conf_spread_acc * 100, 1) AS high_conf_spread_acc_pct,
    mp.high_conf_spread_games,
    ROUND(mp.high_conf_total_acc * 100, 1) AS high_conf_total_acc_pct,
    mp.high_conf_total_games,
    
    mp.spread_calibration,
    mp.total_calibration,
    
    -- SITUATIONAL PERFORMANCE
    ROUND(mp.division_spread_acc * 100, 1) AS division_spread_acc_pct,
    ROUND(mp.cold_weather_total_acc * 100, 1) AS cold_weather_total_acc_pct,
    ROUND(mp.close_game_spread_acc * 100, 1) AS close_game_spread_acc_pct,
    
    -- FEATURE IMPORTANCE INSIGHTS
    ROUND(fp.large_epa_diff_acc * 100, 1) AS large_epa_diff_acc_pct,
    ROUND(fp.small_epa_diff_acc * 100, 1) AS small_epa_diff_acc_pct,
    ROUND(fp.weather_game_acc * 100, 1) AS weather_impact_acc_pct,
    
    -- BETTING ROI ANALYSIS
    ROUND(br.spread_win_rate * 100, 1) AS spread_win_rate_pct,
    ROUND(br.total_win_rate * 100, 1) AS total_win_rate_pct,
    ROUND(br.spread_profit_per_100_games, 0) AS spread_roi_per_100_games,
    ROUND(br.total_profit_per_100_games, 0) AS total_roi_per_100_games,
    
    -- HIGH CONFIDENCE BETTING ANALYSIS
    ROUND(br.high_conf_spread_rate * 100, 1) AS high_conf_spread_win_pct,
    br.high_conf_spread_opportunities,
    ROUND(br.high_conf_total_rate * 100, 1) AS high_conf_total_win_pct,
    br.high_conf_total_opportunities,
    
    -- PROFITABILITY FLAGS
    CASE 
        WHEN br.spread_win_rate > 0.5238 THEN 'Profitable'
        WHEN br.spread_win_rate > 0.50 THEN 'Break_Even'
        ELSE 'Losing'
    END AS spread_betting_status,
    
    CASE 
        WHEN br.total_win_rate > 0.5238 THEN 'Profitable'
        WHEN br.total_win_rate > 0.50 THEN 'Break_Even' 
        ELSE 'Losing'
    END AS total_betting_status,
    
    -- MODEL RECOMMENDATIONS
    CASE 
        WHEN br.high_conf_spread_rate > 0.60 AND br.high_conf_spread_opportunities >= 20 
        THEN 'Focus_High_Confidence_Spreads'
        WHEN mp.enhanced_spread_accuracy > mp.enhanced_total_accuracy 
        THEN 'Focus_Spreads'
        WHEN mp.enhanced_total_accuracy > 0.53 
        THEN 'Focus_Totals'
        ELSE 'Model_Needs_Improvement'
    END AS betting_recommendation

FROM model_performance mp
LEFT JOIN feature_performance fp ON mp.season = fp.season
LEFT JOIN betting_roi br ON mp.season = br.season
ORDER BY mp.season DESC