//+------------------------------------------------------------------+
//| WaveletThinIndicator.mq5                                         |
//| MT5 Thin Indicator — main chart overlay (Trend line)             |
//|                                                                  |
//| Responsibilities (post Architecture Fix):                        |
//|   - Read Trend values from Global Variables (written by EA)      |
//|   - Draw Trend line on main chart                                |
//|   - Display connection status from EA                            |
//|                                                                  |
//| No HTTP. No JSON. No WebRequest.                                 |
//| All data is provided by WaveletBridgeEA via Global Variables.   |
//|                                                                  |
//| Requires WaveletBridgeEA.mq5 to be running on the same chart.   |
//+------------------------------------------------------------------+
#property copyright   "Wavelet Research"
#property version     "2.00"
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   1

// Main chart: Trend line
#property indicator_label1  "Trend"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

//--- Global Variable keys (must match WaveletBridgeEA)
#define GV_STATUS      "Wv_Status"
#define GV_LAST_UPDATE "Wv_LastUpdate"
#define GV_LATENCY     "Wv_Latency"
#define GV_TREND_N     "Wv_Trend_N"
#define GV_TREND_MODE  "Wv_TrendMode"
#define GV_PREFIX      "Wv_"

//--- Input parameters
input bool InpDrawTrend    = true;   // Draw Trend Line
input bool InpShowStatus   = true;   // Show status comment

//--- Indicator buffers
double BufferTrend[];

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufferTrend, INDICATOR_DATA);
   ArraySetAsSeries(BufferTrend, true);

   IndicatorSetString(INDICATOR_SHORTNAME, "WaveletTrend");

   if (InpShowStatus)
      Comment("WaveletTrend: waiting for WaveletBridgeEA...");

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
         Comment("WaveletTrend: EA disconnected — waiting");
      return 0;
   }

   //--- Read how many values the EA stored
   int n = (int)GlobalVariableGet(GV_TREND_N);
   if (n <= 0)
      return 0;

   int limit = MathMin(n, rates_total);

   //--- Fill indicator buffer from Global Variables
   //    GV index 0 = most recent tick = BufferTrend[0]
   if (InpDrawTrend)
   {
      for (int i = 0; i < limit; i++)
      {
         string key = GV_PREFIX + "Trend_" + IntegerToString(i);
         if (GlobalVariableCheck(key))
            BufferTrend[i] = GlobalVariableGet(key);
      }
   }

   //--- Status comment
   if (InpShowStatus)
   {
      double lat       = GlobalVariableGet(GV_LATENCY);
      double mode_num  = GlobalVariableCheck(GV_TREND_MODE)
                         ? GlobalVariableGet(GV_TREND_MODE) : 2.0;
      string mode_str  = (mode_num == 1.0) ? "A1" :
                         (mode_num == 3.0) ? "A3" :
                         (mode_num == 4.0) ? "A4" : "A2";
      string info = "WaveletTrend\nConnected\nMode : " + mode_str
                  + "\nLatency : " + DoubleToString(lat, 1) + " ms";
      Comment(info);
   }

   return rates_total;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Comment("");
}
//+------------------------------------------------------------------+
