{-# LANGUAGE FlexibleContexts #-}


import           Test.Tasty (TestTree, testGroup)
import qualified Test.Tasty as Tasty
import qualified Test.Tasty.SmallCheck as SC

import qualified Data.Map.Strict as M

import qualified Numeric.Backprop as BP

import qualified Net.Util as U
-- import qualified Net.Graph as N
import qualified Net.Graph.Arc as B
import qualified Net.Graph.Marginals as Margs


main :: IO ()
main = Tasty.defaultMain tests


tests :: TestTree
tests = testGroup "Tests" [properties] --, unitTests]


properties :: TestTree
properties = testGroup "Properties" [scProps] -- , qcProps]


scProps = testGroup "(checked by SmallCheck)"
--   [ SC.testProperty "sort == sort . reverse" $
--       \list -> sort (list :: [Int]) == sort (reverse list)
  [ SC.testProperty "rightInTwo (xs ++ xs) == (xs, xs)" $
      \xs -> U.rightInTwo (xs ++ xs :: [Int]) == (xs, xs)
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "(decode . encode) x == x" $
      \x -> (B.decode . B.encode) x == x
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "(explicate . obfuscate) x == x" $
      \xs ->
        let m = M.fromList (zip B.enumerate $ pad 8 xs)
         in null xs || (B.explicate . B.obfuscate) m == m
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "obfuscate == evalBP0 . obfuscateBP" $
      \xs ->
        let m = M.fromList (zip B.enumerate $ pad 8 xs)
            lst = U.toList . B.unVec
            eq v1 v2 = lst v1 == lst v2
         in null xs || B.obfuscate m `eq`
              BP.evalBP0 (B.obfuscateBP (fmap BP.auto m))
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "mask (enumerate !! x) ! y == [x == y]" $
      \x0 y0 ->
        let x = x0 `mod` 8
            y = y0 `mod` 8
         in U.toList (B.mask (B.enumerate !! x)) !! y ==
              if x == y then 1.0 else 0.0
  , SC.testProperty "explicate (mask x) M.! y == [x == y]" $
      \x y -> B.explicate (B.Vec $ B.mask x) M.! y ==
        if x == y then 1.0 else 0.0
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "Res: additive identity" $
      \x -> x + 0 == (x :: Margs.Res Int)
  , Tasty.localOption (SC.SmallCheckDepth 4) .
    SC.testProperty "Res: multiplicative identity" $
      \x -> x * 1 == (x :: Margs.Res Int)
  , Tasty.localOption (SC.SmallCheckDepth 2) .
    SC.testProperty "Res: distributive law" $
      \x y z -> x * (y + z :: Margs.Res Int) == x*y + x*z
  , Tasty.localOption (SC.SmallCheckDepth 2) .
    SC.testProperty "Res: associative multiplication" $
      \x y z -> x * (y * z :: Margs.Res Int) == (x * y) * z
  ]


-- | Pad the given (non-empty) list to the given number of elements.
pad :: Int -> [a] -> [a]
pad k = take k . cycle
