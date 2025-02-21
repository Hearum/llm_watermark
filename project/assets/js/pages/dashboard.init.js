!(function (e) {
  "use strict";
  function a() {}
  (a.prototype.createAreaChart = function (e, a, r, t, i, o, b, s) {
    Morris.Area({
      element: e,
      pointSize: 0,
      lineWidth: 1,
      data: t,
      xkey: i,
      ykeys: o,
      labels: b,
      resize: !0,
      gridLineColor: "rgba(108, 120, 151, 0.1)",
      hideHover: "auto",
      lineColors: s,
      fillOpacity: 0.9,
      behaveLikeLine: !0,
      parseTime: false // 添加此行以禁用时间解析
    });
  });  
    (a.prototype.createDonutChart = function (e, a, r) {
      Morris.Donut({ element: e, data: a, resize: !0, colors: r });
    }),
    (a.prototype.createStackedChart = function (e, a, r, t, i, o) {
      Morris.Bar({
        element: e,
        data: a,
        xkey: r,
        ykeys: t,
        stacked: !0,
        labels: i,
        hideHover: "auto",
        resize: !0,
        gridLineColor: "rgba(108, 120, 151, 0.1)",
        barColors: o,
      });
    }),
    e("#sparkline").sparkline([8, 6, 4, 7, 10, 12, 7, 4, 9, 12, 13, 11, 12], {
      type: "bar",
      height: "130",
      barWidth: "10",
      barSpacing: "7",
      barColor: "#7A6FBE",
    }),
    (a.prototype.init = function () {
      
      this.createAreaChart(
        "morris-area-example",
        0,
        0,
        [
          { y: "00:00", a: 10, b: 15, c: 5 },
          { y: "01:00", a: 20, b: 25, c: 10 },
          { y: "02:00", a: 30, b: 20, c: 8 },
          { y: "03:00", a: 40, b: 8, c: 4 },
          { y: "04:00", a: 50, b: 12, c: 6 },
          { y: "05:00", a: 68, b: 22, c: 9 },
          { y: "06:00", a: 70, b: 30, c: 19 },
          { y: "07:00", a: 80, b: 35, c: 20 },
          { y: "08:00", a: 90, b: 47, c: 25 },
          { y: "09:00", a: 85, b: 52, c: 30 },
          { y: "10:00", a: 70, b: 62, c: 35 },
          { y: "11:00", a: 60, b: 23, c: 40 },
          { y: "12:00", a: 55, b: 32, c: 45 },
          { y: "13:00", a: 42, b: 74, c: 50 },
          { y: "14:00", a: 27, b: 81, c: 65 },
          { y: "15:00", a: 12, b: 42, c: 60 },
          { y: "16:00", a: 22, b: 20, c: 74 },
          { y: "17:00", a: 25, b: 25, c: 83 },
          { y: "18:00", a: 20, b: 22, c: 54 },
          { y: "19:00", a: 65, b: 25, c: 42 },
          { y: "20:00", a: 20, b: 11, c: 33 },
          { y: "21:00", a: 45, b: 35, c: 30 },
          { y: "22:00", a: 30, b: 20, c: 25 },
          { y: "23:00", a: 25, b: 45, c: 20 }
        ],
        "y",
        ["a", "b", "c"],
        ["SQL注入", "XSS攻击", "DDOS攻击"],
        ["#ccc", "#7a6fbe", "#28bbe3"]
      );
      (a.prototype.init = function () {
        this.createAreaChart(
          "morris-area-example",
          0,
          0,
          [
            { y: "00:00", a: 10, b: 15, c: 5 },
            { y: "01:00", a: 20, b: 25, c: 10 },
            { y: "02:00", a: 15, b: 20, c: 8 },
            { y: "03:00", a: 5, b: 8, c: 4 },
            { y: "04:00", a: 8, b: 12, c: 6 },
            { y: "05:00", a: 18, b: 22, c: 9 },
            { y: "06:00", a: 28, b: 30, c: 15 },
            { y: "07:00", a: 40, b: 35, c: 20 },
            { y: "08:00", a: 50, b: 45, c: 25 },
            { y: "09:00", a: 65, b: 55, c: 30 },
            { y: "10:00", a: 75, b: 60, c: 35 },
            { y: "11:00", a: 80, b: 65, c: 40 },
            { y: "12:00", a: 85, b: 70, c: 45 },
            { y: "13:00", a: 90, b: 75, c: 50 },
            { y: "14:00", a: 95, b: 80, c: 55 },
            { y: "15:00", a: 100, b: 85, c: 60 },
            { y: "16:00", a: 90, b: 80, c: 55 },
            { y: "17:00", a: 85, b: 75, c: 50 },
            { y: "18:00", a: 80, b: 70, c: 45 },
            { y: "19:00", a: 75, b: 65, c: 40 },
            { y: "20:00", a: 70, b: 60, c: 35 },
            { y: "21:00", a: 65, b: 55, c: 30 },
            { y: "22:00", a: 60, b: 50, c: 25 },
            { y: "23:00", a: 55, b: 45, c: 20 }
          ],
          "y",
          ["a", "b", "c"],
          ["Series A", "Series B", "Series C"],
          ["#ccc", "#7a6fbe", "#28bbe3"]
        );
      }),      
      this.createDonutChart(
        "morris-donut-example",
        // [
        //     { label: "macOS", value: 160 },
        //     { label: "Windows", value: 140 },
        //     { label: "Linux", value: 65 },
        //     { label: "iOS", value: 32 },
        //     { label: "SafeLine-CE", value: 28 },
        //     { label: "Chrome", value: 124 },
        // ],
        // ["#f0f1f4", "#7a6fbe", "#28bbe3","#34c38f", "#f46a6a", "#556ee6"]
            [
            { label: "macOS", value: 600 },
            { label: "Windows", value: 2500 },
            { label: "Android", value: 90 },
            { label: "Linux", value: 65 },
            { label: "iOS", value: 3 },
            { label: "SafeLine-CE", value: 10 },
            { label: "Chrome", value: 1200 },
            { label: "Edge", value: 357 },
            { label: "curl", value: 64 },
            { label: "Headless Chrome", value: 50 },
            { label: "Core", value: 39 },
            { label: "Trident", value: 29 },
            { label: "Bingbot", value: 25 },
            { label: "Firefox", value: 17 },
            { label: "NetType", value: 15 },
            { label: "Googlebot", value: 14 },
            { label: "Huawei Browser", value: 14 },
            { label: "Wget", value: 13 },
            { label: "Axel", value: 10 },
            { label: "Safari", value: 8 },
            { label: "DuckDuckBot", value: 6 },
            { label: "Internet Explorer", value: 5 },
            { label: "AhrefsBot", value: 4 },
            { label: "Baiduspider", value: 4 },
            { label: "Dalvik", value: 4 },
            { label: "tid", value: 4 },
            { label: "InternetMeasurement", value: 3 },
            { label: "CensysInspect", value: 3 },
            { label: "OAI-SearchBot", value: 2 },
            { label: "YandexBot", value: 2 },
            { label: "Bytespider", value: 2 },
            { label: "Opera", value: 2 },
            { label: "DingTalkBot-LinkService", value: 2 },
            { label: "QuarkPC", value: 2 },
            { label: "ChatGPT-User", value: 2 },
            { label: "Googlebot-Image", value: 1 },
            { label: "DingTalkBot-SecurityService", value: 1 },
            { label: "Mozilla/5.0 zgrab/0.x", value: 1 },
            { label: "NetcraftSurveyAgent", value: 1 },
            { label: "okhttp", value: 1 },
            { label: "PetalBot", value: 1 },
            { label: "Sogou web spider", value: 1 },
            { label: "Baiduspider-render", value: 1 },
            { label: "Yahoo! Slurp", value: 1 },
            { label: "Gecko", value: 1 }
        ], ["#f0f1f4", "#7a6fbe", "#28bbe3", "#34c38f", "#f46a6a", "#556ee6", "#50a5f1", "#ffb822", "#34bfa3", "#e74c3c", "#8e44ad", "#2ecc71", "#3498db", "#f1c40f", "#e67e22", "#16a085", "#d35400", "#2c3e50", "#bdc3c7", "#7f8c8d"]);
        
      this.createStackedChart(
        "morris-bar-stacked",
        [
          { y: "10.27", a: 45, b: 180 },
          { y: "10.28", a: 75, b: 65 },
          { y: "10.29", a: 100, b: 90 },
          { y: "10.30", a: 75, b: 65 },
          { y: "11.1", a: 100, b: 90 },
          { y: "11.2", a: 75, b: 65 },
          { y: "11.3", a: 50, b: 40 },
          { y: "11.4", a: 75, b: 65 },
          { y: "11.5", a: 50, b: 40 },
          { y: "11.6", a: 75, b: 65 },
          { y: "11.7", a: 100, b: 90 },
          { y: "11.8", a: 80, b: 65 },
        ],
        "y",
        ["a", "b"],
        ["放行流量", "拦截流量"],
        ["#28bbe3", "#f0f1f4"]
      );
    }),
    (e.Dashboard = new a()),
    (e.Dashboard.Constructor = a);
})(window.jQuery),
  (function () {
    "use strict";
    window.jQuery.Dashboard.init();
  })();
