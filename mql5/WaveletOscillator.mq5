//+------------------------------------------------------------------+
//| WaveletOscillator.mq5                                            |
//| MT5 Oscillator — separate window for deviation/z-score/energy   |
//|                                                                  |
//| Responsibilities (post Architecture Fix):                        |
//|   - Read RelDev, ZScore, Energy from Global Variables (EA)       |
//|   - Render oscillator lines in a separate chart window           |
//|   - Display connection status from EA                            |
//|                                                                  |
//| No HTTP. No JSON. No WebRequest.                                 |
//| All data is provided by WaveletBridgeEA via Global Variables.   |
//|                                                                  |
//| Requires WaveletBridgeEA.mq5 to be running on the same chart.   |
//+------------------------------------------------------------------+
#property copyright   "Wavelet Research"
#property version     "2.00"
#property indicator_separate_window
#property indicator_buffers 3
#property indicator_plots   3

// Buffer 0: Relative Deviation
#property indicator_label1  "Rel Deviation"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrOrangeRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1

// Buffer 1: Z-Score
#property indicator_label2  "Z-Score"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrMediumPurple
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1

// Buffer 2: Energy
#property indicator_label3  "Energy"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrGold
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1

//--- Global Variable keys (must match WaveletBridgeEA)
#define GV_STATUS    "Wv_Status"
#define GV_LATENCY   "Wv_Latency"
#define GV_TREND_N   "Wv_Trend_N"
#define GV_PREFIX    "Wv_"

//--- Input parameters
input bool InpDrawRelDeviation = true;   // Draw Relative Deviation
input bool InpDrawZScore       = true;   // Draw Z-Score
input bool InpDrawEnergy       = true;   // Draw Energy
input bool InpShowStatus       = true;   // Show status comment

//--- Indicator buffers
double BufferRelDev[];
double BufferZScore[];
double BufferEnergy[];

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufferRelDev, INDICATOR_DATA);
   SetIndexBuffer(1, BufferZScore, INDICATOR_DATA);
   SetIndexBuffer(2, BufferEnergy, INDICATOR_DATA);

   ArraySetAsSeries(BufferRelDev, true);
   ArraySetAsSeries(BufferZScore, true);
   ArraySetAsSeries(BufferEnergy, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "WaveletOsc");

   if (InpShowStatus)
      Comment("WaveletOsc: waiting for WaveletBridgeEA...");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
int OnCalculate(const int       rates_total,
                const int       prev_calculated,
                const datetime& time[],
                const double&   open[],
                const double&   high[],
                const double&   low[],
                const double&   close[],
                const long&     tick_volume[],
                const long&     volume[],
                const int&      spread[])
{
   //--- Read connection status from EA
   bool connected = (GlobalVariableGet(GV_STATUS) >= 1.0);

   if (!connected)
   {
      if (InpShowStatus)
         Comment("WaveletOsc: EA disconnected — waiting");
      return 0;
   }

   //--- Read how many values the EA stored
   int n = (int)GlobalVariableGet(GV_TREND_N);
   if (n <= 0)
      return 0;

   int limit = MathMin(n, rates_total);

   //--- Fill oscillator buffers from Global Variables
   //    GV index 0 = most recent tick = Buffer[0]
   for (int i = 0; i < limit; i++)
   {
      string si = IntegerToString(i);

      if (InpDrawRelDeviation)
      {
         string key_rd = GV_PREFIX + "RelDev_" + si;
         if (GlobalVariableCheck(key_rd))
            BufferRelDev[i] = GlobalVariableGet(key_rd);
      }
      if (InpDrawZScore)
      {
         string key_zs = GV_PREFIX + "ZScore_" + si;
         if (GlobalVariableCheck(key_zs))
            BufferZScore[i] = GlobalVariableGet(key_zs);
      }
      if (InpDrawEnergy)
      {
         string key_en = GV_PREFIX + "Energy_" + si;
         if (GlobalVariableCheck(key_en))
            BufferEnergy[i] = GlobalVariableGet(key_en);
      }
   }

   //--- Status comment
   if (InpShowStatus)
   {
      double lat  = GlobalVariableGet(GV_LATENCY);
      string info = "WaveletOsc: connected | n=" + IntegerToString(n)
                  + " | lat=" + DoubleToString(lat, 1) + "ms";
      Comment(info);
   }

   return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) { Comment(""); }
//+------------------------------------------------------------------+
